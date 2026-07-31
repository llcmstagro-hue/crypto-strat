"""
MT5-тариф против прежней модели: меняет ли он чей-то вердикт.

Сравнение на ОДНОМ И ТОМ ЖЕ конфиге под тремя моделями издержек — иначе
эффект тарифа не отделить от эффекта подбора параметров:

    легаси   6.00 bps/сторона + 9 слиппедж = 30.0 bps round-trip
    MT5      3.25 bps/сторона + 9 слиппедж = 24.5 bps  (замер по счёту)
    MT5 x2   3.25 bps/сторона + 18 слиппедж = 42.5 bps (стресс)

Третья строка тут не для красоты. Разница между первыми двумя — всего
5.5 bps round-trip, и если чей-то вердикт от неё переворачивается, то этот
вердикт по определению стоит на тонком льду. Стресс отвечает на вопрос
«насколько тонком»: 9 bps проскальзывания — оценка, а торговля идёт через
CFD-фид проп-фирмы, где реальность скорее хуже биржевой.

Свежесть считается под КАЖДОЙ моделью отдельно: вопрос трейдера ровно про
это — оживает ли эдж в 2024–2026 при более низкой комиссии.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from crypto_strat.data.loader import load_basket
from crypto_strat.engine.backtest import run_backtest
from crypto_strat.engine.config import LEGACY_COSTS, MT5_COSTS, StrategyConfig

FRESH = pd.Timestamp("2024-01-01", tz="UTC")
RECENT = pd.Timestamp("2025-01-01", tz="UTC")
MT5_X2 = {**MT5_COSTS, "slip_bps_per_side": 18.0}


def _with(cfg: StrategyConfig, costs: dict) -> StrategyConfig:
    d = cfg.to_dict()
    d["costs"] = dict(costs)
    return StrategyConfig.from_dict(d)


def _run(cfg, ds) -> pd.DataFrame:
    out = []
    for s, d in ds.items():
        r = run_backtest(d["ohlcv"], cfg, s, funding=d.get("funding"))
        if r.n_trades:
            t = r.trades.copy()
            t["symbol"] = s
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def concentration(R: np.ndarray) -> float:
    """Доля прибыли, которую дают 5 лучших сделок. Красный флаг р.4."""
    wins = R[R > 0]
    if wins.sum() <= 0:
        return 1.0
    return float(np.sort(wins)[-5:].sum() / wins.sum())


def stats(t: pd.DataFrame) -> dict:
    if not len(t):
        return {}
    R = t["R"].to_numpy()
    f24 = t.loc[t.entry_dt >= FRESH, "R"]
    f25 = t.loc[t.entry_dt >= RECENT, "R"]
    return {"n": len(t), "exp": float(R.mean()),
            "f24": float(f24.mean()) if len(f24) else float("nan"),
            "f25": float(f25.mean()) if len(f25) else float("nan"),
            "n24": len(f24), "n25": len(f25),
            "conc": concentration(R)}


def main() -> int:
    rows = []
    for tf in ("4h", "1d"):
        ds = load_basket("./data", tf=tf)
        with open(f"lab_mt5_{tf}.json", encoding="utf-8") as f:
            rep = json.load(f)
        for r in rep["results"]:
            if not r.get("config"):
                continue
            cfg = StrategyConfig.from_dict(r["config"])
            legacy = stats(_run(_with(cfg, LEGACY_COSTS), ds))
            mt5 = stats(_run(_with(cfg, MT5_COSTS), ds))
            x2 = stats(_run(_with(cfg, MT5_X2), ds))
            rows.append({"name": r["name"], "tf": tf,
                         "barriers": r["barriers"],
                         "robustness": r["robustness"],
                         "failed": r["failed"],
                         "legacy": legacy, "mt5": mt5, "x2": x2,
                         "years": {}})
            t = _run(_with(cfg, MT5_COSTS), ds)
            yr = t.assign(y=t.entry_dt.dt.year).groupby("y")["R"].agg(["size", "mean"])
            rows[-1]["years"] = {int(y): {"n": int(v["size"]), "exp": float(v["mean"])}
                                 for y, v in yr.iterrows()}
            print(f"  посчитано: {r['name']}")

    rows.sort(key=lambda x: (-(x["barriers"] or 0), -x["robustness"]))

    print("\n" + "=" * 126)
    print("MT5-ТАРИФ (24.5 bps) ПРОТИВ ПРЕЖНЕЙ МОДЕЛИ (30.0 bps) — один и тот же конфиг")
    print("=" * 126)
    print(f"{'кандидат':<44}{'ТФ':<4}{'сделок':>7}{'нетто 30bps':>12}"
          f"{'нетто 24.5':>11}{'прирост':>9}{'конц.':>7}{'барьеров':>9}")
    for r in rows:
        d = r["mt5"]["exp"] - r["legacy"]["exp"]
        print(f"{r['name'][:43]:<44}{r['tf']:<4}{r['mt5']['n']:>7}"
              f"{r['legacy']['exp']:>+12.3f}{r['mt5']['exp']:>+11.3f}{d:>+9.3f}"
              f"{r['mt5']['conc']*100:>6.0f}%{str(r['barriers'])+'/7':>9}")

    print("\n" + "-" * 126)
    print("СВЕЖЕСТЬ ЭДЖА: переворачивает ли тариф вердикт (нетто с 2024 и с 2025)")
    print(f"{'кандидат':<44}{'2024+ 30bps':>12}{'2024+ MT5':>11}"
          f"{'2025+ 30bps':>12}{'2025+ MT5':>11}{'2025+ x2слип':>13}  вердикт")
    for r in rows:
        L, M, X = r["legacy"], r["mt5"], r["x2"]
        was = L["f24"] > 0 and L["f25"] > 0
        now = M["f24"] > 0 and M["f25"] > 0
        rob = X["f24"] > 0 and X["f25"] > 0
        v = ("ПЕРЕВОРОТ: мёртв -> жив" if (now and not was) else
             ("жив на обеих моделях" if now else "мёртв на обеих"))
        if now and not rob:
            v += " (но не переживает x2 слиппеджа)"
        print(f"{r['name'][:43]:<44}{L['f24']:>+12.3f}{M['f24']:>+11.3f}"
              f"{L['f25']:>+12.3f}{M['f25']:>+11.3f}{X['f25']:>+13.3f}  {v}")

    flipped = [r for r in rows
               if (r["mt5"]["f24"] > 0 and r["mt5"]["f25"] > 0)
               and not (r["legacy"]["f24"] > 0 and r["legacy"]["f25"] > 0)]
    print("\n" + "-" * 126)
    print(f"вердикт перевернулся у {len(flipped)} из {len(rows)}")
    for r in rows:
        if r["barriers"] and r["barriers"] >= 6:
            print(f"\n  {r['name']} ({r['tf']}): {r['barriers']}/7, "
                  f"робастность {r['robustness']:.3f}, "
                  f"провален только: {', '.join(r['failed'])}")
            print("    нетто по годам (MT5): " + "  ".join(
                f"{y}:{v['exp']:+.2f}({v['n']})" for y, v in sorted(r["years"].items())))
            print(f"    свежесть MT5: 2024+ {r['mt5']['f24']:+.3f}R (n={r['mt5']['n24']}) | "
                  f"2025+ {r['mt5']['f25']:+.3f}R (n={r['mt5']['n25']})")
            print(f"    при удвоении слиппеджа: 2024+ {r['x2']['f24']:+.3f}R | "
                  f"2025+ {r['x2']['f25']:+.3f}R")
            print(f"    концентрация прибыли: {r['mt5']['conc']*100:.0f}% в топ-5 сделках")

    with open("mt5_analysis.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\nОтчёт: mt5_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
