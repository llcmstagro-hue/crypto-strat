"""
Затухание эджа у методов Куртни Смита: 2017–2021 против 2022–2026.

Это ГЛАВНЫЙ вопрос прогона, а не приложение к нему. Все предыдущие заходы
(наши пробои, волатильность, межрынок, ансамбли, Conqueror) показали одно и
то же затухание. Возражение к такому выводу всегда одно: «вы проверяли свои
формализации». Методы из книги Смита — чужие правила, придуманные до крипты
и под другой рынок; если и они затухают, вывод перестаёт быть свойством
нашего кода.

Тест — ПЕРЕСТАНОВОЧНЫЙ, а не t-критерий. Причина конкретная: распределение R
у трендовых систем чудовищно правохвостое (одна сделка по DOGE в 2021 даёт
+623R), нормальность не выполняется даже приблизительно, и t-критерий на
таких данных врёт в обе стороны. Перестановка не требует ничего, кроме
обмениваемости меток эпох при нулевой гипотезе.

Отдельно печатается МЕДИАНА и доля прибыльных: если среднее падает, а медиана
нет, значит ушли не сделки, а хвост — и это другая новость.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from crypto_strat.data.loader import load_basket
from crypto_strat.engine.backtest import run_backtest
from crypto_strat.engine.config import StrategyConfig

SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
N_PERM = 20000
SEED = 20260729


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM) -> float:
    """Двусторонний перестановочный тест на разницу средних.

    p считается с поправкой (+1)/(+1) — так оценка не может выдать ровно 0
    и не завышает значимость на конечном числе перестановок.
    """
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def era_stats(r: np.ndarray) -> dict:
    if not len(r):
        return {"n": 0}
    wins = r[r > 0].sum()
    loss = -r[r < 0].sum()
    return {"n": len(r), "exp": float(r.mean()), "med": float(np.median(r)),
            "wr": float((r > 0).mean()),
            "pf": float(wins / loss) if loss > 0 else float("inf")}


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "lab_smith_1d.json"
    with open(src, encoding="utf-8") as f:
        report = json.load(f)
    ds = load_basket("./data", tf="1d")

    rows = []
    for r in report["results"]:
        if not r.get("config"):
            continue
        cfg = StrategyConfig.from_dict(r["config"])
        trades = []
        for s, d in ds.items():
            bt = run_backtest(d["ohlcv"], cfg, s, funding=d.get("funding"))
            if bt.n_trades:
                trades.append(bt.trades[["entry_dt", "R"]])
        if not trades:
            continue
        t = pd.concat(trades, ignore_index=True)
        old = t.loc[t["entry_dt"] < SPLIT, "R"].to_numpy()
        new = t.loc[t["entry_dt"] >= SPLIT, "R"].to_numpy()
        if len(old) < 30 or len(new) < 30:
            print(f"{r['name']}: пропущен, эпоха слишком мала "
                  f"({len(old)}/{len(new)})")
            continue
        a, b = era_stats(old), era_stats(new)
        p = permutation_p(old, new)
        rows.append({"name": r["name"], "type": r["type"],
                     "barriers": r["barriers"], "old": a, "new": b, "p": p,
                     "drop": (a["exp"] - b["exp"]) / abs(a["exp"])
                             if a["exp"] else float("nan")})
        print(f"  посчитано: {r['name']}")

    rows.sort(key=lambda x: -x["barriers"])
    print("\n" + "=" * 108)
    print("ЗАТУХАНИЕ ЭДЖА: 2017–2021 -> 2022–2026 (методы Куртни Смита, D1)")
    print("=" * 108)
    print(f"{'метод':<32}{'бар':>4}{'n стар':>8}{'exp стар':>10}{'n нов':>7}"
          f"{'exp нов':>10}{'спад':>8}{'p':>9}  медиана стар->нов")
    for x in rows:
        a, b = x["old"], x["new"]
        print(f"{x['name'][:31]:<32}{x['barriers']:>3}/7{a['n']:>8}"
              f"{a['exp']:>+10.3f}{b['n']:>7}{b['exp']:>+10.3f}"
              f"{x['drop']*100:>7.0f}%{x['p']:>9.4f}"
              f"  {a['med']:+.3f} -> {b['med']:+.3f}")

    sig = [x for x in rows if x["p"] < 0.05]
    dec = [x for x in rows if x["new"]["exp"] < x["old"]["exp"]]
    print("-" * 108)
    print(f"методов проверено: {len(rows)}; эдж ниже в 2022–2026: {len(dec)}; "
          f"падение значимо (p<0.05): {len(sig)}")
    if rows:
        print(f"максимальный p среди значимых: "
              f"{max((x['p'] for x in sig), default=float('nan')):.4f}")

    with open("smith_decay.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("Отчёт: smith_decay.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
