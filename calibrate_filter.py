"""
Этап 4 — КАЛИБРОВКА ФИЛЬТРА. Проверка ИНСТРУМЕНТА, не стратегий.

Зачем это обязательно до поиска: фильтр, который никого не пропускает,
и фильтр, который пропускает всех, одинаково бесполезны, и по одному прогону
их не отличить. Нужен эталон с ЗАРАНЕЕ ИЗВЕСТНЫМ ответом.

Матрица калибровки:

  A. ЧУВСТВИТЕЛЬНОСТЬ  — честная стратегия на данных, где эдж ТОЧНО ЕСТЬ.
                          Ожидание: ПРОХОДИТ. Не прошла -> фильтр глухой,
                          он зарежет и настоящие находки.

  B. СПЕЦИФИЧНОСТЬ     — та же честная стратегия на ЧИСТОМ ШУМЕ.
                          Ожидание: ОТСЕВ. Прошла -> в фильтре дыра
                          (утечка будущего или заниженные пороги).

  C. АНТИ-ПОДГОНКА     — заведомо переподогнанные версии на данных с эджем:
                          микроскопический стоп и «шесть фильтров на малой
                          выборке». Ожидание: ОТСЕВ по конкретным барьерам.

Фильтр считается откалиброванным, только если A, B и C дали ожидаемый ответ.
Если фильтр НЕ РАЗЛИЧАЕТ — работа останавливается, дальше идти нельзя.

Запуск:
    python calibrate_filter.py                 # синтетика (нет реальных данных)
    python calibrate_filter.py --data ./data   # реальные CSV, когда появятся
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from crypto_strat.data.loader import load_basket
from crypto_strat.data.synthetic import make_basket
from crypto_strat.engine.config import StrategyConfig
from crypto_strat.validation.barriers import Thresholds, thresholds_for_tf
from crypto_strat.validation.filter import run_filter, robustness_score
from crypto_strat.validation.hypothesis import Hypothesis, TrialLog

SYNTH_WARNING = (
    "ДАННЫЕ СИНТЕТИЧЕСКИЕ. Метрики стратегий отсюда НЕ доказывают эдж — "
    "они доказывают только поведение фильтра. Калибровку обязательно "
    "повторить на реальных CSV."
)


# --------------------------------------------------------------------------- #
# ГИПОТЕЗЫ КАЛИБРОВКИ
# --------------------------------------------------------------------------- #
def honest_breakout() -> Hypothesis:
    """ЧЕСТНАЯ стратегия — позитивный контроль.

    Простая, мало правил (вход + стоп + выход), стоп в ATR (не подсвечный),
    узкая осмысленная сетка. Именно такую фильтр обязан пропускать, когда
    в данных есть эдж.
    """
    cfg = StrategyConfig.from_dict({
        "name": "honest_donchian",
        "direction": "both",
        "entry": {"type": "donchian_breakout", "params": {"period": 20}},
        "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
        "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0,
                                                    "max_bars": 240}},
        "filters": [],
        "sizing": {"risk_pct": 0.01},
        "meta": {"source": "калибровка: позитивный контроль"},
    })
    return Hypothesis(cfg, grid={
        "entry.params.period": [15, 20, 30, 40],
        "stop.params.mult": [1.5, 2.0, 2.5],
        "exit.params.mult": [2.5, 3.0, 3.5],
    }, source="калибровка / позитивный контроль", grid_cap=40)


def honest_order_block() -> Hypothesis:
    """ЧЕСТНЫЙ order block — прямой порт smc_order_block_backtest.py.

    Правила без изменений: BOS по подтверждённым свингам, зона = последняя
    противоположная свеча, вход лимитом при возврате, стоп за зоной, тейк по R/R.
    """
    cfg = StrategyConfig.from_dict({
        "name": "honest_order_block",
        "direction": "both",
        "entry": {"type": "order_block", "params": {
            "swing_left": 3, "swing_right": 3, "ob_use_body": False,
            "ob_max_age": 60, "stop_buffer": 0.0005}},
        "stop": {"type": "block", "params": {}},
        "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}},
        "filters": [],
        "sizing": {"risk_pct": 0.01},
        "meta": {"source": "smc_order_block_backtest.py, методология трейдера"},
    })
    return Hypothesis(cfg, grid={
        "entry.params.swing_left": [2, 3, 4],
        "entry.params.ob_max_age": [40, 60, 80],
        "exit.params.rr": [1.5, 2.0, 3.0],
    }, source="калибровка / методология трейдера (р.6.5 п.4)", grid_cap=40)


def overfit_tiny_stop(base: str = "donchian") -> Hypothesis:
    """ПОДГОНКА №1: микроскопический стоп ради красивого R/R.

    Стоп 0.15% от цены на H1 — заметно меньше типичной свечи. В бэктесте это
    рисует высокий R/R, вживую его снимает внутрибаровый шум, которого в OHLC
    не видно (р.10 п.3). Плюс на единицу риска приходится гигантский нотионал,
    поэтому 0.3% round-trip превращается в ~2R издержек на сделку.
    Ожидание: смерть на барьере 7 (и, скорее всего, ещё на 1).
    """
    if base == "donchian":
        entry = {"type": "donchian_breakout", "params": {"period": 20}}
        name = "overfit_tinystop_donchian"
    else:
        entry = {"type": "order_block", "params": {"swing_left": 3, "swing_right": 3,
                                                   "ob_max_age": 60}}
        name = "overfit_tinystop_ob"
    cfg = StrategyConfig.from_dict({
        "name": name, "direction": "both", "entry": entry,
        "stop": {"type": "pct", "params": {"pct": 0.0015}},
        "exit": {"type": "fixed_rr", "params": {"rr": 4.0, "max_bars": 120}},
        "filters": [],
        "sizing": {"risk_pct": 0.01},
        "meta": {"source": "калибровка: заведомая подгонка (мелкий стоп)"},
    })
    return Hypothesis(cfg, grid={
        "stop.params.pct": [0.001, 0.0015, 0.002],
        "exit.params.rr": [3.0, 4.0, 5.0],
    }, source="калибровка / заведомая подгонка №1", grid_cap=20)


def overfit_many_filters(base: str = "donchian", has_volume: bool = True) -> Hypothesis:
    """ПОДГОНКА №2: шесть фильтров, вырезающих узкое «удачное» подмножество.

    Каждый фильтр по отдельности звучит разумно — вместе они оставляют горстку
    сделок, на которой любая метрика становится шумом (р.4, красные флаги).
    Ожидание: смерть на барьере 4 (минимум сделок) и/или 2.
    """
    entry = ({"type": "donchian_breakout", "params": {"period": 20}} if base == "donchian"
             else {"type": "order_block", "params": {"swing_left": 3, "swing_right": 3,
                                                     "ob_max_age": 60}})
    cfg = StrategyConfig.from_dict({
        "name": f"overfit_manyfilters_{base}",
        "direction": "long",
        "entry": entry,
        "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
        "exit": {"type": "fixed_rr", "params": {"rr": 2.5, "max_bars": 96}},
        "filters": [
            {"type": "htf_trend", "params": {"factor": 4, "ema_period": 50}},
            {"type": "session", "params": {"start_hour": 13, "end_hour": 17}},
            {"type": "atr_regime", "params": {"min_pct": 0.004, "max_pct": 0.011}},
            ({"type": "volume", "params": {"period": 20, "mult": 1.3}} if has_volume
             else {"type": "ma_side", "params": {"period": 200, "kind": "sma"}}),
            {"type": "rsi_bound", "params": {"period": 14, "min": 52, "max": 68}},
            {"type": "ema_slope", "params": {"period": 100, "lag": 20, "min_slope": 0.004}},
        ],
        "sizing": {"risk_pct": 0.01},
        "meta": {"source": "калибровка: заведомая подгонка (много фильтров)"},
    })
    return Hypothesis(cfg, grid={
        "filters.4.params.min": [48, 52, 56],
        "filters.2.params.min_pct": [0.003, 0.004, 0.005],
        "exit.params.rr": [2.0, 2.5, 3.0],
    }, source="калибровка / заведомая подгонка №2", grid_cap=40)


# --------------------------------------------------------------------------- #
# ПРОГОН
# --------------------------------------------------------------------------- #
def load_dataset(data_dir: str | None, regime: str, tf: str, n: int):
    if data_dir and os.path.isdir(data_dir):
        ds = load_basket(data_dir, tf=tf)
        if ds:
            return ds, False
        print(f"! в {data_dir} нет подходящих CSV — падаю на синтетику")
    return make_basket(regime=regime, n=n, tf=tf), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="папка с реальными CSV")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--bars", type=int, default=35_000, help="длина синтетики (~4 года H1)")
    ap.add_argument("--out", default="calibration_report.json")
    args = ap.parse_args()

    th = thresholds_for_tf(args.tf)
    results = {}

    print("=" * 78)
    print("КАЛИБРОВКА ФИЛЬТРА АНТИ-ОВЕРФИТА (Этап 4)")
    print("=" * 78)

    # ---------- A + C: данные С ЭДЖЕМ ---------- #
    ds_edge, synth = load_dataset(args.data, "edge", args.tf, args.bars)
    note = SYNTH_WARNING if synth else ""
    primary = [list(ds_edge)[0]]
    print(f"\nДанные: {'СИНТЕТИКА (режим edge)' if synth else args.data} | "
          f"инструментов: {len(ds_edge)} | подбор параметров на: {primary[0]}")
    if synth:
        print(f"⚠️  {SYNTH_WARNING}")

    tl_edge = TrialLog("./trials_calibration_edge.json")
    tl_edge.reset()

    hv = bool(next(iter(ds_edge.values()))["ohlcv"].attrs.get("has_volume", True))
    if not hv:
        print("ℹ️  в данных нет объёма — объёмный фильтр в подгонке C2/C4 заменён "
              "на ценовой (ma_side), число правил сохранено")

    plan = [
        ("A. чувствительность", honest_breakout(), True,
         "честная стратегия на данных с эджем -> ДОЛЖНА ПРОЙТИ"),
        ("A'. order block (как просили)", honest_order_block(), None,
         "честный OB трейдера; проходит или нет — зависит от того, есть ли "
         "ЕГО эдж в этих данных"),
        ("C1. подгонка: мелкий стоп", overfit_tiny_stop("donchian"), False,
         "должна умереть на издержках/стопе"),
        ("C2. подгонка: много фильтров", overfit_many_filters("donchian", hv), False,
         "должна умереть на минимуме сделок"),
        ("C3. подгонка OB: мелкий стоп", overfit_tiny_stop("ob"), False,
         "то же на блоке трейдера"),
        ("C4. подгонка OB: много фильтров", overfit_many_filters("ob", hv), False,
         "то же на блоке трейдера"),
    ]

    for label, hypo, expect_pass, why in plan:
        print(f"\n{'-'*78}\n{label}  [{hypo.name}]\n  ожидание: {why}\n{'-'*78}")
        v = run_filter(ds_edge, hypo, primary=primary, th=th, trial_log=tl_edge,
                       fail_fast=False, data_note=note)
        print(v.report())
        results[label] = _pack(v, expect_pass)

    # ---------- B: ЧИСТЫЙ ШУМ ---------- #
    print(f"\n{'='*78}\nB. СПЕЦИФИЧНОСТЬ — та же честная стратегия на ЧИСТОМ ШУМЕ")
    print("   ожидание: ОТСЕВ. Если пройдёт — в фильтре дыра.\n" + "=" * 78)
    noise_bars = min(len(d["ohlcv"]) for d in ds_edge.values()) if not synth else args.bars
    ds_noise, synth_n = load_dataset(None, "noise", args.tf, noise_bars)
    tl_noise = TrialLog("./trials_calibration_noise.json")
    tl_noise.reset()
    v_noise = run_filter(ds_noise, honest_breakout(), primary=[list(ds_noise)[0]],
                         th=th, trial_log=tl_noise, fail_fast=False,
                         data_note="ЧИСТЫЙ ШУМ: эджа нет по построению")
    print(v_noise.report())
    results["B. специфичность (шум)"] = _pack(v_noise, False)

    # ---------- ВЕРДИКТ ---------- #
    verdict = _verdict(results, real=not synth)
    print("\n" + "=" * 78)
    print("ВЕРДИКТ КАЛИБРОВКИ")
    print("=" * 78)
    for line in verdict["lines"]:
        print(line)
    print("-" * 78)
    print(f"Конфигов перебрано: edge={tl_edge.total}, noise={tl_noise.total}")
    print(f"ИТОГ: {verdict['status']}")
    print("=" * 78)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "verdict": verdict,
                   "synthetic": synth, "trials_edge": tl_edge.total,
                   "trials_noise": tl_noise.total}, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {os.path.abspath(args.out)}")

    return 0 if verdict["calibrated"] else 2


def _pack(v, expect_pass) -> dict:
    return {
        "hypothesis": v.hypothesis,
        "survived": v.survived,
        "expected_pass": expect_pass,
        "barriers_passed": v.n_passed,
        "robustness": round(v.robustness, 4),
        "failed_barriers": [b.name for b in v.barriers if not b.passed],
        "notes": {b.name: b.note for b in v.barriers},
        "metrics": {k: v.metrics.get(k) for k in
                    ("n_trades", "winrate", "profit_factor", "expectancy_R",
                     "sharpe_trade", "max_loss_streak")},
        "flags": v.flags,
        "best_config": v.best_config.name if v.best_config else None,
    }


def _verdict(results: dict, real: bool = False) -> dict:
    """Вердикт калибровки.

    ВАЖНО про тест A на РЕАЛЬНЫХ данных: он перестаёт быть критерием.
    Смысл теста чувствительности — «честная стратегия на данных, где эдж ТОЧНО
    ЕСТЬ, обязана пройти». На реальном рынке никто не знает, есть ли там эдж,
    поэтому «не прошла» одинаково объясняется и глухим фильтром, и отсутствием
    эджа — а различить эти два случая нечем. Ground truth есть только у
    синтетики, поэтому чувствительность калибруется там, а на реальных данных
    A остаётся справочным и критериями работают B (специфичность) и C
    (анти-подгонка).
    """
    lines, ok = [], True

    a = results.get("A. чувствительность", {})
    if real:
        lines.append(f"ℹ️  A. ЧУВСТВИТЕЛЬНОСТЬ: на реальных данных НЕ является "
                     f"критерием (истинный ответ неизвестен). Факт: "
                     f"{'прошла' if a.get('survived') else 'отсев'} "
                     f"({a.get('barriers_passed')}/7).")
    elif a.get("survived"):
        lines.append("✅ A. ЧУВСТВИТЕЛЬНОСТЬ: честная стратегия на данных с эджем ПРОШЛА.")
    else:
        ok = False
        lines.append("❌ A. ЧУВСТВИТЕЛЬНОСТЬ: честная стратегия НЕ прошла "
                     f"(упала на: {', '.join(a.get('failed_barriers', []))}). "
                     "Фильтр глухой — он зарежет и настоящие находки.")

    b = results.get("B. специфичность (шум)", {})
    if not b.get("survived"):
        lines.append("✅ B. СПЕЦИФИЧНОСТЬ: на чистом шуме стратегия ОТСЕЯНА "
                     f"(барьеры: {', '.join(b.get('failed_barriers', []))}).")
    else:
        ok = False
        lines.append("❌ B. СПЕЦИФИЧНОСТЬ: стратегия прошла фильтр НА ЧИСТОМ ШУМЕ. "
                     "Это дыра: утечка будущего или заниженные пороги.")

    overfits = {k: v for k, v in results.items() if k.startswith("C")}
    survived = [k for k, v in overfits.items() if v.get("survived")]
    if not survived:
        lines.append(f"✅ C. АНТИ-ПОДГОНКА: все {len(overfits)} переподогнанных "
                     "версий отсеяны.")
        for k, v in overfits.items():
            lines.append(f"     - {k}: упала на {', '.join(v.get('failed_barriers', [])) or '—'}")
    else:
        ok = False
        lines.append(f"❌ C. АНТИ-ПОДГОНКА: подгонка ПРОШЛА фильтр: {survived}")

    ob = results.get("A'. order block (как просили)", {})
    if ob:
        lines.append(f"ℹ️  A'. order block трейдера: "
                     f"{'ПРОШЁЛ' if ob.get('survived') else 'отсев'} "
                     f"({ob.get('barriers_passed')}/7; упал на: "
                     f"{', '.join(ob.get('failed_barriers', [])) or '—'})")

    return {"calibrated": ok, "lines": lines,
            "status": ("ФИЛЬТР ОТКАЛИБРОВАН — можно идти к поиску"
                       if ok else
                       "ФИЛЬТР НЕ РАЗЛИЧАЕТ — СТОП, дальше идти нельзя")}


if __name__ == "__main__":
    sys.exit(main())
