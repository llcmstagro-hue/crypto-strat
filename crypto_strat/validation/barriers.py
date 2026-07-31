"""
Барьеры анти-оверфита (р.4). СЕРДЦЕ СИСТЕМЫ.

Стратегия проходит дальше, только если переживает ВСЕ семь. Провал любого = отсев.
Барьеры подобраны так, чтобы их нельзя было пройти перебором:

  1 OOS               — метрика меряется на данных, которых оптимизатор не видел
  2 Walk-forward      — эдж обязан держаться в БОЛЬШИНСТВЕ окон, не в одном
  3 Кросс-инструмент  — эдж обязан жить не только там, где его нашли
  4 Минимум сделок    — меньше порога метрика = шум, красота не спасает
  5 Multiple testing  — планка растёт с числом проб (deflated Sharpe)
  6 Стабильность      — эдж живёт на ПЛАТО параметров, а не на игле
  7 Издержки          — round-trip >= 0.3%, funding, запрет подсвечных стопов

Ключевой принцип отбора параметров внутри барьеров: целевая функция — НЕ
доходность. Оптимизируем Sharpe по сделкам с жёстким гейтом по их числу
(р.7: оптимизировать под робастность, не под целевые метрики).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..engine.backtest import run_backtest, BacktestResult
from ..engine.config import StrategyConfig, MIN_ROUND_TRIP_BPS, MIN_MAKER_FEE_BPS, MIN_TAKER_FEE_BPS, MIN_SLIP_BPS
from ..engine import indicators as ind
from ..engine.metrics import trade_metrics, full_metrics
from .hypothesis import Hypothesis, TrialLog
from . import stats as st

MS_DAY = 86_400_000


# --------------------------------------------------------------------------- #
# ПОРОГИ — все в одном месте, чтобы их было видно и можно было оспорить
# --------------------------------------------------------------------------- #
@dataclass
class Thresholds:
    # барьер 4
    min_trades_total: int = 200
    min_trades_per_symbol: int = 30
    min_symbols_supporting: int = 3
    # барьер 1
    oos_min_trades: int = 50
    oos_min_expectancy: float = 0.0
    oos_min_pf: float = 1.05
    oos_min_retention: float = 0.30      # test_sharpe >= 0.30 * train_sharpe
    # барьер 2
    wf_min_windows: int = 4
    wf_min_share_positive: float = 0.60
    wf_train_days: int = 365
    wf_test_days: int = 90
    # барьер 3
    cross_min_positive: int = 3
    cross_catastrophe_R: float = -0.25   # ниже этого — «катастрофа» на инструменте
    special_symbol: str = "DOGEUSDT"     # ходит на настроениях, особый режим (р.4)
    # барьер 5
    dsr_min: float = 0.95
    # барьер 6
    stab_min_plateau: float = 0.50       # медиана соседей >= 50% от центра
    stab_min_frac_positive: float = 0.60
    # барьер 7
    min_stop_to_atr: float = 0.50        # стоп не мельче половины ATR данного ТФ
    cost_stress_mult: float = 2.0
    # красные флаги
    max_rules: int = 6


def thresholds_for_tf(tf: str) -> Thresholds:
    """Пороги под плотность данных таймфрейма.

    Меняется ТОЛЬКО длина окон walk-forward — это параметр ПРОЦЕДУРЫ, а не
    критерий прохождения. Все пороги «пройдено/не пройдено» (минимум сделок,
    DSR, доля положительных окон, плато, издержки) одинаковы на всех ТФ.

    Зачем это нужно. На D1 окно train в 365 дней даёт 8-11 сделок при
    минимуме отбора в 30. Тогда КАЖДЫЙ внутренний перебор возвращает -inf,
    подбор молча откатывается к дефолтному конфигу, и барьер walk-forward
    перестаёт что-либо проверять — при этом выглядит пройденным или нет
    по совершенно посторонним причинам. Расширение окна возвращает барьеру
    смысл; ценой становится меньшее число окон, и оно по-прежнему обязано
    быть >= wf_min_windows.
    """
    th = Thresholds()
    if tf == "1d":
        th.wf_train_days = 1095     # 3 года: ~30 сделок на подбор
        th.wf_test_days = 365
    elif tf == "4h":
        th.wf_train_days = 365
        th.wf_test_days = 90
    return th


@dataclass
class BarrierResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    note: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.note}"


# --------------------------------------------------------------------------- #
# ПРОГОНЫ
# --------------------------------------------------------------------------- #
def run_symbols(dataset: dict, cfg: StrategyConfig, symbols=None,
                ts_from: int | None = None, ts_to: int | None = None,
                initial_equity: float = 25_000.0) -> list[BacktestResult]:
    """Прогон конфига по набору инструментов в заданном временном окне."""
    out = []
    for sym in (symbols or list(dataset)):
        d = dataset.get(sym)
        if d is None:
            continue
        df = d["ohlcv"]
        if ts_from is not None or ts_to is not None:
            m = np.ones(len(df), dtype=bool)
            if ts_from is not None:
                m &= df["ts"].to_numpy() >= ts_from
            if ts_to is not None:
                m &= df["ts"].to_numpy() < ts_to
            df = df[m].reset_index(drop=True)
            # attrs теряются при нарезке — переносим вручную. refs НЕ режем:
            # межрыночные блоки выравнивают ведущий инструмент по таймстампу,
            # поэтому лишние бары впереди безвредны, а обрезка сзади сломала бы
            # причинность окна
            for key in ("tf", "has_volume", "refs"):
                if key in d["ohlcv"].attrs:
                    df.attrs[key] = d["ohlcv"].attrs[key]
        if len(df) < 200:
            continue
        out.append(run_backtest(df, cfg, sym, funding=d.get("funding"),
                                initial_equity=initial_equity))
    return out


def pooled_R(results: list[BacktestResult]) -> np.ndarray:
    arrs = [r.R for r in results if r.n_trades]
    return np.concatenate(arrs) if arrs else np.array([], dtype=float)


def objective(results: list[BacktestResult], min_trades: int = 30) -> float:
    """Целевая функция подбора параметров.

    Sharpe по сделкам, а НЕ доходность и НЕ profit factor. Почему так:
      * доходность — прямая дорога к подгонке под целевые 3%/мес (р.0, р.7);
      * PF взрывается на малых выборках (пара крупных выигрышей -> PF 5);
      * Sharpe по сделкам штрафует разброс, то есть меряет устойчивость.
    Выборка меньше min_trades отбрасывается совсем: там метрика — шум.
    """
    R = pooled_R(results)
    if len(R) < min_trades:
        return -np.inf
    m = trade_metrics(R)
    return m["sharpe_trade"]


def grid_search(dataset: dict, hypo: Hypothesis, symbols: list[str],
                ts_from: int | None, ts_to: int | None,
                trial_log: TrialLog | None = None, tag: str = "",
                selection: bool = True) -> tuple:
    """Подбор параметров ТОЛЬКО на переданном окне (обычно — train).

    Возвращает (лучший конфиг, таблица всех проб). Таблица нужна барьеру 5:
    из разброса Sharpe по пробам считается, насколько высоко мог залететь
    победитель по чистой случайности.
    """
    variants = hypo.variants()
    rows = []
    for cfg in variants:
        res = run_symbols(dataset, cfg, symbols, ts_from, ts_to)
        sc = objective(res)
        R = pooled_R(res)
        rows.append({"cfg": cfg, "score": sc, "n_trades": len(R),
                     "expectancy": float(R.mean()) if len(R) else 0.0})
    if trial_log is not None:
        trial_log.record(len(variants), tag or hypo.name, selection=selection)

    finite = [r for r in rows if np.isfinite(r["score"])]
    # Ни один вариант не набрал минимума сделок -> выбирать не из чего, и мы
    # молча откатываемся к дефолтному конфигу. На разреженных данных (D1) это
    # превращает подбор в фикцию, поэтому факт отмечается явно, а не проглатывается.
    degraded = not finite
    best = max(finite, key=lambda r: r["score"]) if finite else None
    for r in rows:
        r["degraded"] = degraded
    return (best["cfg"] if best else hypo.config), rows


# --------------------------------------------------------------------------- #
# БАРЬЕР 1 — OUT-OF-SAMPLE
# --------------------------------------------------------------------------- #
def barrier_oos(dataset: dict, hypo: Hypothesis, primary: list[str],
                th: Thresholds, split: float = 0.70,
                trial_log: TrialLog | None = None) -> tuple[BarrierResult, StrategyConfig, list]:
    """Параметры подбираются на train, метрика меряется на test.

    Сплит строго ПО ВРЕМЕНИ. Перемешивать ряд нельзя: случайный сплит пустит
    будущее в обучение и покажет эдж там, где его нет.
    """
    ts_all = np.concatenate([dataset[s]["ohlcv"]["ts"].to_numpy() for s in primary])
    t0, t1 = int(ts_all.min()), int(ts_all.max())
    t_split = t0 + int((t1 - t0) * split)

    best_cfg, table = grid_search(dataset, hypo, primary, t0, t_split, trial_log, f"{hypo.name}/oos")

    tr_res = run_symbols(dataset, best_cfg, primary, t0, t_split)
    te_res = run_symbols(dataset, best_cfg, primary, t_split, t1 + 1)
    tr = trade_metrics(pooled_R(tr_res))
    te = trade_metrics(pooled_R(te_res))

    retention = (te["sharpe_trade"] / tr["sharpe_trade"]
                 if tr["sharpe_trade"] > 0 else (1.0 if te["sharpe_trade"] > 0 else 0.0))

    checks = {
        "трейдов на test": te["n_trades"] >= th.oos_min_trades,
        "ожидаемость test > 0": te["expectancy_R"] > th.oos_min_expectancy,
        "PF test": te["profit_factor"] >= th.oos_min_pf,
        "эдж не осыпался": retention >= th.oos_min_retention,
    }
    passed = all(checks.values())
    note = (f"train exp={tr['expectancy_R']:+.3f}R sharpe={tr['sharpe_trade']:+.3f} "
            f"n={tr['n_trades']} -> test exp={te['expectancy_R']:+.3f}R "
            f"PF={te['profit_factor']:.2f} sharpe={te['sharpe_trade']:+.3f} "
            f"n={te['n_trades']} (сохранение эджа {retention*100:.0f}%)")
    return (BarrierResult("1. Out-of-sample", passed,
                          {"train": tr, "test": te, "retention": retention,
                           "checks": checks, "split_ts": t_split}, note),
            best_cfg, table)


# --------------------------------------------------------------------------- #
# БАРЬЕР 2 — WALK-FORWARD
# --------------------------------------------------------------------------- #
def barrier_walkforward(dataset: dict, hypo: Hypothesis, primary: list[str],
                        th: Thresholds, trial_log: TrialLog | None = None) -> BarrierResult:
    """Катящееся окно: train N мес -> test M мес -> сдвиг -> повтор.

    В КАЖДОМ окне параметры подбираются заново, только на его train. Это и
    отвечает на вопрос «разваливается ли стратегия при смене режима рынка»:
    один удачный период тут не спасает, нужно большинство окон.
    """
    ts_all = np.concatenate([dataset[s]["ohlcv"]["ts"].to_numpy() for s in primary])
    t0, t1 = int(ts_all.min()), int(ts_all.max())
    train_ms = th.wf_train_days * MS_DAY
    test_ms = th.wf_test_days * MS_DAY

    windows, start = [], t0
    while start + train_ms + test_ms <= t1:
        windows.append((start, start + train_ms, start + train_ms + test_ms))
        start += test_ms

    if len(windows) < th.wf_min_windows:
        return BarrierResult(
            "2. Walk-forward", False,
            {"n_windows": len(windows), "need": th.wf_min_windows},
            f"истории хватает лишь на {len(windows)} окон "
            f"(нужно >= {th.wf_min_windows}) — судить не о чем")

    rows, oos_R = [], []
    for (a, b, c) in windows:
        cfg, tbl = grid_search(dataset, hypo, primary, a, b, trial_log,
                               f"{hypo.name}/wf", selection=False)
        res = run_symbols(dataset, cfg, primary, b, c)
        R = pooled_R(res)
        m = trade_metrics(R)
        oos_R.append(R)
        rows.append({
            "from": pd.to_datetime(b, unit="ms", utc=True).date().isoformat(),
            "to": pd.to_datetime(c, unit="ms", utc=True).date().isoformat(),
            "n": m["n_trades"], "expectancy_R": m["expectancy_R"],
            "sharpe": m["sharpe_trade"], "cfg": cfg.name,
            "degraded": bool(tbl and tbl[0].get("degraded")),
        })

    # окна без сделок не считаем «положительными», но и в знаменатель берём:
    # стратегия, которая полгода не торгует, тоже не проходит
    positive = sum(1 for r in rows if r["n"] > 0 and r["expectancy_R"] > 0)
    share = positive / len(rows)
    allR = np.concatenate([r for r in oos_R if len(r)]) if any(len(r) for r in oos_R) else np.array([])
    pooled = trade_metrics(allR)

    checks = {
        "большинство окон в плюсе": share >= th.wf_min_share_positive,
        "суммарный OOS в плюсе": pooled["expectancy_R"] > 0,
    }
    passed = all(checks.values())
    n_degraded = sum(1 for r in rows if r.get("degraded"))
    checks["подбор в окнах не выродился"] = n_degraded <= len(rows) // 2
    passed = all(checks.values())
    note = (f"{positive}/{len(rows)} окон в плюсе ({share*100:.0f}%), "
            f"суммарно OOS exp={pooled['expectancy_R']:+.3f}R n={pooled['n_trades']}"
            + (f"; ⚠️ в {n_degraded}/{len(rows)} окнах сделок не хватило на подбор "
               f"(взят дефолтный конфиг)" if n_degraded else ""))
    return BarrierResult("2. Walk-forward", passed,
                         {"windows": rows, "share_positive": share,
                          "degraded_windows": n_degraded,
                          "pooled": pooled, "checks": checks}, note)


# --------------------------------------------------------------------------- #
# БАРЬЕР 3 — КРОСС-ИНСТРУМЕНТ
# --------------------------------------------------------------------------- #
def barrier_cross_instrument(dataset: dict, cfg: StrategyConfig,
                             th: Thresholds, primary: list[str]) -> tuple[BarrierResult, list]:
    """ФИКСИРОВАННЫЙ конфиг гоняется по всей корзине.

    Подчёркнуто без переподбора параметров под каждый инструмент: смысл барьера
    именно в том, что один и тот же эдж должен работать на разных рынках.
    Переподбор превратил бы проверку в пять новых подгонок.

    DOGE — особый случай (р.4): ходит на настроениях, не на структуре. Он не
    обязан подтверждать эдж, но КАТАСТРОФЫ на нём быть не должно.
    """
    results = run_symbols(dataset, cfg, list(dataset))
    per = []
    for r in results:
        m = trade_metrics(r.R)
        per.append({"symbol": r.symbol, "n": m["n_trades"],
                    "expectancy_R": m["expectancy_R"], "pf": m["profit_factor"],
                    "sharpe": m["sharpe_trade"], "primary": r.symbol in primary})

    eligible = [p for p in per if p["n"] >= th.min_trades_per_symbol]
    ordinary = [p for p in eligible if p["symbol"] != th.special_symbol]
    positive = [p for p in ordinary if p["expectancy_R"] > 0]

    special = next((p for p in per if p["symbol"] == th.special_symbol), None)
    special_ok = True
    if special and special["n"] >= th.min_trades_per_symbol:
        special_ok = special["expectancy_R"] > th.cross_catastrophe_R

    checks = {
        "хватает инструментов с выборкой": len(eligible) >= th.min_symbols_supporting,
        "эдж держится на большинстве": len(positive) >= th.cross_min_positive,
        f"нет катастрофы на {th.special_symbol}": special_ok,
    }
    passed = all(checks.values())
    note = (f"в плюсе {len(positive)}/{len(ordinary)} инструментов с выборкой "
            f"(всего пригодных {len(eligible)}/{len(per)})"
            + (f"; {th.special_symbol}: exp={special['expectancy_R']:+.3f}R n={special['n']}"
               if special else ""))
    return BarrierResult("3. Кросс-инструмент", passed,
                         {"per_symbol": per, "checks": checks}, note), results


# --------------------------------------------------------------------------- #
# БАРЬЕР 4 — МИНИМУМ СДЕЛОК
# --------------------------------------------------------------------------- #
def barrier_min_trades(per_symbol: list[dict], th: Thresholds) -> BarrierResult:
    """Жёсткий порог выборки. Ни одна красивая метрика не отменяет статистику:
    на 40 сделках PF 3.0 не отличим от везения."""
    total = sum(p["n"] for p in per_symbol)
    enough = [p for p in per_symbol if p["n"] >= th.min_trades_per_symbol]
    checks = {
        f"всего сделок >= {th.min_trades_total}": total >= th.min_trades_total,
        f"инструментов с >= {th.min_trades_per_symbol} сделок": len(enough) >= th.min_symbols_supporting,
    }
    passed = all(checks.values())
    note = (f"всего {total} сделок; инструментов с достаточной выборкой "
            f"{len(enough)}/{len(per_symbol)}")
    return BarrierResult("4. Минимум сделок", passed,
                         {"total": total, "eligible": len(enough), "checks": checks}, note)


# --------------------------------------------------------------------------- #
# БАРЬЕР 5 — MULTIPLE TESTING
# --------------------------------------------------------------------------- #
def barrier_multiple_testing(oos_results: list[BacktestResult], table: list,
                             trial_log: TrialLog, th: Thresholds) -> BarrierResult:
    """Deflated Sharpe: планка тем выше, чем больше гипотез перебрано.

    n_trials берётся ГЛОБАЛЬНЫЙ (все конфиги, прогнанные за всю историю
    проекта), а не по одной гипотезе. Это принципиально: если перебрать
    50 гипотез по 20 конфигов, победитель — лучший из 1000, и планка
    обязана соответствовать 1000, а не 20.
    """
    R = pooled_R(oos_results)
    m = trade_metrics(R)
    sharpes = [r["score"] for r in table if np.isfinite(r["score"])]
    var = st.sr_variance_across_trials(sharpes)
    # пул ОТБОРА, а не общая нагрузка (см. TrialLog)
    n_trials = max(trial_log.selection, len(table), 1)

    d = st.deflated_sharpe_ratio(m["sharpe_trade"], m["n_trades"], m["skew"],
                                 m["kurtosis"], n_trials, var)
    mtrl = st.min_track_record_length(m["sharpe_trade"], m["skew"], m["kurtosis"],
                                      sr_benchmark=d["sr_threshold"])
    passed = d["dsr"] >= th.dsr_min
    note = (f"DSR={d['dsr']:.3f} (нужно >= {th.dsr_min}) при {n_trials} пробах отбора; "
            f"SR={d['sr']:+.3f} против планки случайного максимума "
            f"{d['sr_threshold']:+.3f}; нужно сделок для значимости: "
            + ("бесконечно" if not np.isfinite(mtrl) else f"{mtrl:.0f}")
            + f", есть {m['n_trades']}")
    return BarrierResult("5. Поправка на multiple testing", passed,
                         {**d, "min_track_record": mtrl, "metrics": m}, note)


# --------------------------------------------------------------------------- #
# БАРЬЕР 6 — СТАБИЛЬНОСТЬ ПАРАМЕТРОВ
# --------------------------------------------------------------------------- #
def barrier_param_stability(dataset: dict, hypo: Hypothesis, cfg: StrategyConfig,
                            primary: list[str], th: Thresholds,
                            ts_from: int | None = None, ts_to: int | None = None,
                            trial_log: TrialLog | None = None) -> BarrierResult:
    """Соседи по сетке должны давать близкий результат.

    Проверяется на TRAIN-окне, а не на OOS: тратить out-of-sample на диагностику
    нельзя, он одноразовый. Если PF=3 при периоде 55 и разваливается при 54 и 56 —
    это подгонка под шум, а не эдж.
    """
    nb = hypo.neighbours(cfg)
    if not nb:
        return BarrierResult("6. Стабильность параметров", False, {},
                             "у гипотезы нет числовой сетки — плато проверить нечем")

    base = run_symbols(dataset, cfg, primary, ts_from, ts_to)
    base_m = trade_metrics(pooled_R(base))
    center = base_m["expectancy_R"]

    rows, n_evals = [], 0
    for path, values in nb.items():
        for v in values:
            alt = cfg.with_param(path, v)
            res = run_symbols(dataset, alt, primary, ts_from, ts_to)
            m = trade_metrics(pooled_R(res))
            rows.append({"path": path, "value": v, "n": m["n_trades"],
                         "expectancy_R": m["expectancy_R"], "sharpe": m["sharpe_trade"]})
            n_evals += 1
    if trial_log is not None:
        trial_log.record(n_evals, f"{hypo.name}/stability", selection=False)

    exps = np.array([r["expectancy_R"] for r in rows])
    frac_pos = float((exps > 0).mean()) if len(exps) else 0.0
    plateau = float(np.median(exps) / center) if center > 0 else 0.0

    checks = {
        "медиана соседей держится": plateau >= th.stab_min_plateau,
        "большинство соседей в плюсе": frac_pos >= th.stab_min_frac_positive,
        "центр положительный": center > 0,
    }
    passed = all(checks.values())
    note = (f"центр exp={center:+.3f}R, медиана соседей={np.median(exps):+.3f}R "
            f"({plateau*100:.0f}% от центра), в плюсе {frac_pos*100:.0f}% "
            f"из {len(rows)} соседей")
    return BarrierResult("6. Стабильность параметров", passed,
                         {"center_expectancy": center, "plateau_ratio": plateau,
                          "frac_positive": frac_pos, "neighbours": rows,
                          "checks": checks}, note)


# --------------------------------------------------------------------------- #
# БАРЬЕР 7 — РЕАЛИСТИЧНЫЕ ИЗДЕРЖКИ
# --------------------------------------------------------------------------- #
def barrier_costs(dataset: dict, cfg: StrategyConfig, results: list[BacktestResult],
                  th: Thresholds) -> BarrierResult:
    """Три проверки в одной.

    (a) round-trip >= 0.3% нотионала + funding — иначе тест нечестен by design;
    (b) стоп не мельче половины ATR своего ТФ. Стоп меньше размера свечи на
        старшем ТФ ЧЕСТНО НЕ ТЕСТИРУЕТСЯ (р.10 п.3): в бэктесте он выглядит
        как источник высокого R/R, а вживую его выбивает внутрибаровый шум,
        которого в OHLC просто не видно;
    (c) стресс: удвоенные издержки. Эдж, исчезающий при удвоении комиссии,
        это не эдж, а зазор в модели исполнения.
    """
    # round-trip считается ПО ФАКТУ исполнений, а не по названию модели:
    # доля мейкерных входов известна только после прогона, и у одной и той же
    # стратегии она может быть не 0 и не 1.
    fee_sum = sum(float(r.trades["fees"].sum()) for r in results if r.n_trades)
    notional_sum = sum(float(r.trades["notional"].sum())
                       for r in results if r.n_trades)
    rt = (10_000.0 * fee_sum / notional_sum) if notional_sum > 0 else cfg.round_trip_bps()
    maker_share = 0.0
    n_tr = sum(r.n_trades for r in results)
    if n_tr:
        maker_share = sum(float((r.trades["entry_kind"] == "limit").sum())
                          for r in results if r.n_trades) / n_tr

    if cfg.is_flat():
        cost_ok = rt >= MIN_ROUND_TRIP_BPS
    else:
        # В модели maker/taker единого порога быть не может: у лимитного входа
        # честный round-trip физически ниже, чем у рыночного. Полы стоят на
        # КОМПОНЕНТАХ и проверены конструктором конфига, здесь остаётся
        # убедиться, что они не подменены на ходу.
        cost_ok = (float(cfg.costs["maker_fee_bps"]) >= MIN_MAKER_FEE_BPS - 1e-9
                   and float(cfg.costs["taker_fee_bps"]) >= MIN_TAKER_FEE_BPS - 1e-9
                   and float(cfg.costs["slip_bps"]) >= MIN_SLIP_BPS - 1e-9)

    # (b) стоп против ATR
    ratios = []
    for r in results:
        if not r.n_trades:
            continue
        d = dataset.get(r.symbol)
        if d is None:
            continue
        df = d["ohlcv"]
        a = ind.atr(df["high"].to_numpy(), df["low"].to_numpy(),
                    df["close"].to_numpy(), 14) / df["close"].to_numpy()
        med_atr = float(np.nanmedian(a))
        med_stop = float(r.trades["stop_dist_pct"].median())
        if med_atr > 0:
            ratios.append(med_stop / med_atr)
    stop_ratio = float(np.median(ratios)) if ratios else 0.0
    stop_ok = stop_ratio >= th.min_stop_to_atr

    # (c) стресс по издержкам
    stressed = cfg.to_dict()
    if cfg.is_flat():
        stressed["costs"]["fee_bps_per_side"] *= th.cost_stress_mult
        stressed["costs"]["slip_bps_per_side"] *= th.cost_stress_mult
    else:
        # Удваиваем и комиссию, и проскальзывание. Комиссия задана биржей и
        # вырасти вдвое не может — но стресс здесь не про прогноз тарифа, а
        # про то, остаётся ли эдж, если модель исполнения ошибается вдвое.
        for k_ in ("maker_fee_bps", "taker_fee_bps", "slip_bps"):
            stressed["costs"][k_] *= th.cost_stress_mult
    stressed["name"] = cfg.name + "|cost_x2"
    s_res = run_symbols(dataset, StrategyConfig.from_dict(stressed), list(dataset))
    s_m = trade_metrics(pooled_R(s_res))
    stress_ok = s_m["expectancy_R"] > 0

    base_m = trade_metrics(pooled_R(results))
    cost_share = 0.0
    fees = sum(float(r.trades["fees"].sum()) + float(r.trades["funding"].sum())
               for r in results if r.n_trades)
    gross = sum(float(r.trades["gross_pnl"].sum()) for r in results if r.n_trades)
    if gross:
        cost_share = fees / abs(gross)

    cost_label = (f"round-trip >= {MIN_ROUND_TRIP_BPS} bps" if cfg.is_flat()
                  else "ставки не ниже реальных (maker 2 / taker 5.5 / слип 9 bps)")
    checks = {
        cost_label: cost_ok,
        "стоп не мельче 0.5 ATR": stop_ok,
        "переживает удвоение издержек": stress_ok,
    }
    passed = all(checks.values())
    note = (f"round-trip {rt:.1f} bps по факту "
            f"(мейкерных входов {maker_share*100:.0f}%), "
            f"издержки съедают {cost_share*100:.0f}% валовой; "
            f"медиана стопа = {stop_ratio:.2f} ATR; при x{th.cost_stress_mult:.0f} "
            f"издержках exp={s_m['expectancy_R']:+.3f}R (было {base_m['expectancy_R']:+.3f}R)")
    return BarrierResult("7. Реалистичные издержки", passed,
                         {"round_trip_bps": rt, "maker_share": maker_share,
                          "stop_to_atr": stop_ratio,
                          "cost_share": cost_share, "stressed": s_m,
                          "checks": checks}, note)


# --------------------------------------------------------------------------- #
# КРАСНЫЕ ФЛАГИ (не барьер — предупреждения в отчёт, р.4)
# --------------------------------------------------------------------------- #
def profit_concentration(results: list[BacktestResult], top_n: int = 5) -> dict:
    """Какая доля прибыли приходится на несколько лучших сделок.

    Зачем отдельная метрика. Ожидаемость в R на трендследующих стратегиях с
    трейлингом/структурным выходом легко улетает в +3R за счёт одной-двух
    парабол (альтсезон 2021 даёт сделки на сотни R). Средняя при этом
    выглядит великолепно, а медиана сидит в минусе. Такая «стратегия» — не
    воспроизводимый эдж, а ставка на повторение конкретного исторического
    события.
    """
    R = pooled_R(results)
    if len(R) < 10:
        return {"top_share": 0.0, "median_R": 0.0, "max_R": 0.0, "n": len(R)}
    gains = R[R > 0]
    if gains.sum() <= 0:
        return {"top_share": 0.0, "median_R": float(np.median(R)),
                "max_R": float(R.max()), "n": len(R)}
    top = np.sort(gains)[-top_n:]
    return {"top_share": float(top.sum() / gains.sum()),
            "median_R": float(np.median(R)),
            "max_R": float(R.max()), "n": len(R), "top_n": top_n}


def red_flags(cfg: StrategyConfig, pooled: dict, per_symbol: list[dict],
              th: Thresholds, concentration: dict | None = None) -> list[str]:
    flags = []
    if concentration and concentration["n"] >= 10:
        cc = concentration
        if cc["top_share"] >= 0.5:
            flags.append(
                f"{cc.get('top_n', 5)} лучших сделок дают {cc['top_share']*100:.0f}% "
                f"всей прибыли (макс. сделка {cc['max_R']:+.0f}R при медиане "
                f"{cc['median_R']:+.2f}R) — результат держится на единичных событиях, "
                f"а не на воспроизводимом эдже")
        elif cc["median_R"] < 0 and pooled["expectancy_R"] > 0.5:
            flags.append(
                f"ожидаемость {pooled['expectancy_R']:+.2f}R при ОТРИЦАТЕЛЬНОЙ медиане "
                f"{cc['median_R']:+.2f}R — среднее тянут редкие крупные выигрыши")
    if cfg.n_rules() > th.max_rules and pooled["n_trades"] < 300:
        flags.append(f"правил {cfg.n_rules()} (> {th.max_rules}) при "
                     f"{pooled['n_trades']} сделках — много условий на малой выборке")
    if pooled["profit_factor"] > 2.5 and pooled["n_trades"] < 100:
        flags.append(f"PF {pooled['profit_factor']:.2f} на {pooled['n_trades']} сделках — "
                     f"высокий PF на малой выборке")
    with_trades = [p for p in per_symbol if p["n"] >= th.min_trades_per_symbol]
    pos = [p for p in with_trades if p["expectancy_R"] > 0]
    if with_trades and len(pos) == 1:
        flags.append(f"эдж есть только на {pos[0]['symbol']} и отсутствует на остальных")
    if pooled["winrate"] > 0.75:
        flags.append(f"winrate {pooled['winrate']*100:.0f}% — проверь, не мелкий ли тейк "
                     f"при большом стопе")
    return flags
