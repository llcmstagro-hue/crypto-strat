"""
Fade ложного пробоя: ведёт ли он себя ИНАЧЕ, чем затухшее трендовое?

Три вопроса, ради которых написан скрипт:

  1. РАЗБИВКА ПО ГОДАМ и значимость 2017–21 -> 2022–26. Главное здесь не
     «затухает ли», а НЕ РАСТЁТ ЛИ. Гипотеза трейдера прямая: если пробои
     стали чаще проваливаться (и это убило трендследование), то ставка ПРОТИВ
     пробоя должна была окрепнуть. Проверяется ОДНОСТОРОННЯЯ версия теста
     тоже — «эдж вырос» — потому что двусторонний p на растущем эффекте
     читается неправильно.

  2. СВЕЖЕСТЬ: положителен ли эдж в 2024–2026.

  3. ВАЛОВОЙ ЭДЖ ОТДЕЛЬНО ОТ ИЗДЕРЖЕК. Это не украшение отчёта, а
     необходимость: у fade стоп стоит вплотную за выбитым хвостом (медиана
     0.5–0.6 ATR), поэтому нотионал на единицу риска огромен, и издержки
     съедают сотни процентов валовой прибыли. Без разделения нельзя отличить
     «сигнал пустой» от «сигнал есть, но не окупает 0.3% round-trip».
     Разные диагнозы -> разные выводы, и второй ещё и лечится (реже сигналы,
     дальше стоп, мейкер вместо тейкера).

Конфиг берётся ОБЪЯВЛЕННЫЙ ЗАРАНЕЕ (R/R 2.0, буфер стопа 0.05%), а не
выбранный сеткой победитель: вопрос стоит об ИДЕЕ, и подставлять сюда
лучший из перебранных вариантов значило бы отвечать на него смещённо.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analyze_smith_decay import permutation_p, SPLIT
from crypto_strat.data.loader import load_basket
from crypto_strat.engine.backtest import run_backtest
from crypto_strat.search.fade import fade_pool

FRESH = pd.Timestamp("2024-01-01", tz="UTC")
RECENT = pd.Timestamp("2025-01-01", tz="UTC")


def collect(cfg, ds) -> pd.DataFrame:
    out = []
    for s, d in ds.items():
        r = run_backtest(d["ohlcv"], cfg, s, funding=d.get("funding"))
        if r.n_trades:
            t = r.trades.copy()
            t["symbol"] = s
            # валовой результат в единицах риска: сколько дал САМ СИГНАЛ,
            # до комиссии и проскальзывания
            t["R_gross"] = t["gross_pnl"] / t["risk"]
            t["cost_R"] = (t["fees"] + t["funding"]) / t["risk"]
            t["edge_pct_notional"] = t["gross_pnl"] / t["notional"]
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def line(tag, t) -> str:
    if not len(t):
        return f"  {tag:<18} нет сделок"
    return (f"  {tag:<18} n={len(t):>5}  нетто={t['R'].mean():+.3f}R  "
            f"валовой={t['R_gross'].mean():+.3f}R  "
            f"издержки={t['cost_R'].mean():.3f}R  "
            f"winrate={(t['R'] > 0).mean()*100:.1f}%")


def one_sided_growth_p(old: np.ndarray, new: np.ndarray,
                       n_perm: int = 20000) -> float:
    """p для ОДНОСТОРОННЕЙ гипотезы «эдж в новой эпохе ВЫШЕ».

    Нужна отдельно: двусторонний тест отвечает «есть ли разница», а вопрос
    трейдера — «не вырос ли». Если p здесь мал, значит рост не случаен.
    """
    obs = new.mean() - old.mean()
    pool = np.concatenate([old, new])
    n_old = len(old)
    rng = np.random.default_rng(4242)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if pool[n_old:].mean() - pool[:n_old].mean() >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main() -> int:
    report = {}
    for tf in ("1d", "4h"):
        ds = load_basket("./data", tf=tf)
        print("\n" + "=" * 104)
        print(f"FADE ЛОЖНОГО ПРОБОЯ — {tf} (конфиг объявлен заранее, не подобран)")
        print("=" * 104)
        for hypo, _idea in fade_pool(tf):
            t = collect(hypo.config, ds)
            if not len(t):
                continue
            print(f"\n{hypo.name}")
            print(line("ВСЯ ИСТОРИЯ", t))
            print(line("2017-21", t[t.entry_dt < SPLIT]))
            print(line("2022-26", t[t.entry_dt >= SPLIT]))
            print(line("с 2024 (свежесть)", t[t.entry_dt >= FRESH]))
            print(line("с 2025 (свежесть)", t[t.entry_dt >= RECENT]))

            old = t.loc[t.entry_dt < SPLIT, "R"].to_numpy()
            new = t.loc[t.entry_dt >= SPLIT, "R"].to_numpy()
            og = t.loc[t.entry_dt < SPLIT, "R_gross"].to_numpy()
            ng = t.loc[t.entry_dt >= SPLIT, "R_gross"].to_numpy()
            p2 = permutation_p(old, new)
            pg = permutation_p(og, ng)
            pgrow = one_sided_growth_p(og, ng)
            print(f"  эпохи: нетто {old.mean():+.3f} -> {new.mean():+.3f} "
                  f"(p дву={p2:.4f}) | валовой {og.mean():+.3f} -> {ng.mean():+.3f} "
                  f"(p дву={pg:.4f}, p «вырос»={pgrow:.4f})")

            years = (t.assign(y=t.entry_dt.dt.year)
                      .groupby("y")
                      .agg(n=("R", "size"), net=("R", "mean"),
                           gross=("R_gross", "mean")))
            print("  по годам (нетто / валовой):")
            print("    " + "  ".join(
                f"{int(y)}:{r.net:+.2f}/{r.gross:+.2f}({int(r.n)})"
                for y, r in years.iterrows()))
            print(f"  медиана стопа: {t['stop_dist_pct'].median()*100:.2f}% цены; "
                  f"валовой эдж на сделку: "
                  f"{t['edge_pct_notional'].mean()*100:+.4f}% нотионала "
                  f"против 0.30% round-trip")
            report[f"{hypo.name}"] = {
                "n": len(t), "net": float(t["R"].mean()),
                "gross": float(t["R_gross"].mean()),
                "cost_R": float(t["cost_R"].mean()),
                "old_net": float(old.mean()), "new_net": float(new.mean()),
                "old_gross": float(og.mean()), "new_gross": float(ng.mean()),
                "p_two_net": p2, "p_two_gross": pg, "p_grew_gross": pgrow,
                "fresh_2024_net": float(t.loc[t.entry_dt >= FRESH, "R"].mean()),
                "fresh_2024_gross": float(t.loc[t.entry_dt >= FRESH, "R_gross"].mean()),
                "fresh_2025_net": float(t.loc[t.entry_dt >= RECENT, "R"].mean()),
                "fresh_2025_gross": float(t.loc[t.entry_dt >= RECENT, "R_gross"].mean()),
                "stop_pct_median": float(t["stop_dist_pct"].median()),
                "edge_pct_notional": float(t["edge_pct_notional"].mean()),
                "years": {int(y): {"n": int(r.n), "net": float(r.net),
                                   "gross": float(r.gross)}
                          for y, r in years.iterrows()},
            }
    with open("fade_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nОтчёт: fade_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
