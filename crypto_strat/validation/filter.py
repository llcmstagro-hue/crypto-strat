"""
Модуль [3] — оркестратор фильтра анти-оверфита.

Прогоняет гипотезу через все семь барьеров р.4 и выносит вердикт.
Выживание = ВСЕ барьеры пройдены. Никаких «прошла шесть из семи, ну почти».

Ранжирование выживших — ПО РОБАСТНОСТИ, не по доходности (р.7). В score
намеренно НЕ входят ни PF, ни ожидаемость, ни доходность: иначе ранжирование
превратится в ту же оптимизацию под метрику, от которой мы защищаемся.
Входит только то, что говорит об устойчивости: доля положительных окон
walk-forward, ширина плато параметров, доля корзины, где эдж жив, сохранение
эджа на OOS и deflated Sharpe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..engine.config import StrategyConfig
from ..engine.metrics import trade_metrics, equity_metrics
from .barriers import (
    Thresholds, BarrierResult, barrier_oos, barrier_walkforward,
    barrier_cross_instrument, barrier_min_trades, barrier_multiple_testing,
    barrier_param_stability, barrier_costs, red_flags, run_symbols, pooled_R,
)
from .hypothesis import Hypothesis, TrialLog


@dataclass
class FilterVerdict:
    hypothesis: str
    survived: bool
    barriers: list[BarrierResult] = field(default_factory=list)
    best_config: StrategyConfig | None = None
    robustness: float = 0.0
    metrics: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    source: str = ""
    elapsed_s: float = 0.0
    data_note: str = ""

    @property
    def n_passed(self) -> int:
        return sum(1 for b in self.barriers if b.passed)

    def report(self) -> str:
        L = [f"{'='*78}",
             f"ГИПОТЕЗА: {self.hypothesis}",
             f"источник: {self.source or '—'}"]
        if self.data_note:
            L.append(f"⚠️  {self.data_note}")
        L.append("-" * 78)
        for b in self.barriers:
            L.append(f"  {'✅' if b.passed else '❌'} {b.name}")
            L.append(f"      {b.note}")
        L.append("-" * 78)
        if self.metrics.get("n_trades"):
            m = self.metrics
            L.append(f"  Метрики (вся корзина, фикс. конфиг): сделок {m['n_trades']}, "
                     f"winrate {m['winrate']*100:.1f}%, PF {m['profit_factor']:.2f}, "
                     f"exp {m['expectancy_R']:+.3f}R, "
                     f"макс. серия убытков {m['max_loss_streak']}")
        if self.flags:
            L.append("  🚩 красные флаги:")
            for f in self.flags:
                L.append(f"      - {f}")
        L.append(f"  ИТОГ: {'ВЫЖИЛА' if self.survived else 'ОТСЕВ'} "
                 f"({self.n_passed}/{len(self.barriers)} барьеров, "
                 f"робастность {self.robustness:.3f}, {self.elapsed_s:.0f}с)")
        L.append("=" * 78)
        return "\n".join(L)


def robustness_score(barriers: list[BarrierResult]) -> float:
    """0..1. Только признаки устойчивости — доходность сюда не допускается."""
    by = {b.name.split(".")[0]: b for b in barriers}
    parts = []

    wf = by.get("2")
    if wf and "share_positive" in wf.detail:
        parts.append(("walk-forward", 0.30, float(np.clip(wf.detail["share_positive"], 0, 1))))

    cr = by.get("3")
    if cr and "per_symbol" in cr.detail:
        per = [p for p in cr.detail["per_symbol"] if p["n"] >= 30]
        share = (sum(1 for p in per if p["expectancy_R"] > 0) / len(per)) if per else 0.0
        parts.append(("кросс-инструмент", 0.25, float(share)))

    stab = by.get("6")
    if stab and "plateau_ratio" in stab.detail:
        pl = float(np.clip(stab.detail["plateau_ratio"], 0, 1))
        fp = float(np.clip(stab.detail["frac_positive"], 0, 1))
        parts.append(("плато параметров", 0.20, 0.5 * pl + 0.5 * fp))

    oos = by.get("1")
    if oos and "retention" in oos.detail:
        parts.append(("сохранение на OOS", 0.15,
                      float(np.clip(oos.detail["retention"], 0, 1))))

    mt = by.get("5")
    if mt and "dsr" in mt.detail:
        parts.append(("deflated Sharpe", 0.10, float(np.clip(mt.detail["dsr"], 0, 1))))

    if not parts:
        return 0.0
    wsum = sum(w for _, w, _ in parts)
    return float(sum(w * v for _, w, v in parts) / wsum)


def run_filter(dataset: dict, hypo: Hypothesis,
               primary: list[str] | None = None,
               th: Thresholds | None = None,
               trial_log: TrialLog | None = None,
               fail_fast: bool = True,
               data_note: str = "",
               verbose: bool = True) -> FilterVerdict:
    """Полный прогон гипотезы через фильтр.

    primary — инструмент(ы), на которых РАЗРЕШЕНО подбирать параметры.
    Остальная корзина остаётся нетронутой и служит независимой проверкой
    (барьер 3). По умолчанию — первый инструмент набора.
    """
    t_start = time.time()
    th = th or Thresholds()
    trial_log = trial_log or TrialLog()
    primary = primary or [list(dataset)[0]]
    trial_log.record_hypothesis()

    verdict = FilterVerdict(hypothesis=hypo.name, survived=False,
                            source=hypo.source, data_note=data_note)

    def log(b: BarrierResult):
        verdict.barriers.append(b)
        if verbose:
            print(f"    {'✅' if b.passed else '❌'} {b.name}: {b.note}")

    # --- 1. OOS (здесь же подбираются параметры — только на train) ---
    b1, best_cfg, table = barrier_oos(dataset, hypo, primary, th, trial_log=trial_log)
    verdict.best_config = best_cfg
    log(b1)
    if fail_fast and not b1.passed:
        return _finish(verdict, t_start)

    split_ts = b1.detail["split_ts"]
    ts_all = np.concatenate([dataset[s]["ohlcv"]["ts"].to_numpy() for s in primary])
    t0, t1 = int(ts_all.min()), int(ts_all.max())

    # --- 6. Стабильность параметров (на train, OOS не тратим) ---
    b6 = barrier_param_stability(dataset, hypo, best_cfg, primary, th,
                                 ts_from=t0, ts_to=split_ts, trial_log=trial_log)
    log(b6)
    if fail_fast and not b6.passed:
        return _finish(verdict, t_start)

    # --- 3. Кросс-инструмент (фиксированный конфиг, вся корзина) ---
    b3, all_results = barrier_cross_instrument(dataset, best_cfg, th, primary)
    log(b3)
    if fail_fast and not b3.passed:
        return _finish(verdict, t_start)

    # --- 4. Минимум сделок ---
    b4 = barrier_min_trades(b3.detail["per_symbol"], th)
    log(b4)
    if fail_fast and not b4.passed:
        return _finish(verdict, t_start)

    # --- 2. Walk-forward (самый дорогой — поэтому не первый) ---
    b2 = barrier_walkforward(dataset, hypo, primary, th, trial_log=trial_log)
    log(b2)
    if fail_fast and not b2.passed:
        return _finish(verdict, t_start)

    # --- 5. Multiple testing на OOS-части ---
    oos_results = run_symbols(dataset, best_cfg, primary, split_ts, t1 + 1)
    b5 = barrier_multiple_testing(oos_results, table, trial_log, th)
    log(b5)
    if fail_fast and not b5.passed:
        return _finish(verdict, t_start)

    # --- 7. Издержки ---
    b7 = barrier_costs(dataset, best_cfg, all_results, th)
    log(b7)

    # --- итог ---
    pooled = trade_metrics(pooled_R(all_results))
    verdict.metrics = pooled
    verdict.flags = red_flags(best_cfg, pooled, b3.detail["per_symbol"], th)
    return _finish(verdict, t_start)


def _finish(v: FilterVerdict, t_start: float) -> FilterVerdict:
    v.survived = len(v.barriers) == 7 and all(b.passed for b in v.barriers)
    v.robustness = robustness_score(v.barriers)
    v.elapsed_s = time.time() - t_start
    return v


def run_batch(dataset: dict, hypotheses: list[Hypothesis], **kw) -> list[FilterVerdict]:
    """Пачка гипотез. Выжившие ранжируются ПО РОБАСТНОСТИ.

    Пустой список выживших — валидный результат (р.7): значит, в проверенном
    пространстве честного эджа нет. Подкручивать пороги, чтобы «хоть что-то
    прошло», запрещено.
    """
    out = []
    for i, h in enumerate(hypotheses, 1):
        print(f"\n[{i}/{len(hypotheses)}] {h.name}  ({h.source})")
        out.append(run_filter(dataset, h, **kw))
    out.sort(key=lambda v: (v.survived, v.robustness), reverse=True)
    return out


def summary_table(verdicts: list[FilterVerdict], trial_log: TrialLog) -> str:
    L = ["", "=" * 92, "СВОДКА ПО ФИЛЬТРУ", "=" * 92,
         f"Гипотез прогнано: {trial_log.hypotheses} | "
         f"конфигов перебрано ВСЕГО: {trial_log.total}",
         "(это число — вход барьера 5: чем больше перебор, тем выше планка победителю)",
         "-" * 92,
         f"{'гипотеза':<34}{'барьеров':>10}{'робастн.':>10}{'сделок':>8}"
         f"{'exp,R':>9}{'PF':>7}  итог"]
    for v in verdicts:
        m = v.metrics
        L.append(f"{v.hypothesis[:33]:<34}{v.n_passed}/7{'':>6}{v.robustness:>10.3f}"
                 f"{m.get('n_trades', 0):>8}{m.get('expectancy_R', 0):>+9.3f}"
                 f"{m.get('profit_factor', 0):>7.2f}  "
                 f"{'ВЫЖИЛА' if v.survived else 'отсев'}")
    survivors = [v for v in verdicts if v.survived]
    L.append("-" * 92)
    if survivors:
        L.append(f"Выживших: {len(survivors)} из {len(verdicts)}")
    else:
        L.append("Выживших НЕТ. Это валидный результат: в проверенном пространстве "
                 "честного эджа не найдено.")
    L.append("=" * 92)
    return "\n".join(L)
