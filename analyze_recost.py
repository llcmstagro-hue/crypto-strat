"""
Перерасчёт на реальных издержках: что изменилось и достаточно ли этого.

Сравнение делается НА ОДНОМ И ТОМ ЖЕ конфиге под двумя моделями издержек —
иначе нельзя отделить эффект тарифа от эффекта подбора параметров. Валовой
результат при этом не меняется вообще (он не зависит от комиссии), меняется
только нетто, и вся разница целиком объясняется моделью исполнения.

Отдельно считается стресс ×2 ПО ПРОСКАЛЬЗЫВАНИЮ выхода. Он важнее обычного
стресса по всем издержкам: комиссия задана биржей и вырасти вдвое не может,
а проскальзывание — наша оценка, и именно в ней может быть ошибка.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from crypto_strat.data.loader import load_basket
from crypto_strat.engine.backtest import run_backtest
from crypto_strat.engine.config import StrategyConfig
from crypto_strat.search.recost import recost_pool
from crypto_strat.search.fade import fade_pool

SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
FRESH = pd.Timestamp("2024-01-01", tz="UTC")
RECENT = pd.Timestamp("2025-01-01", tz="UTC")

LEGACY = {"model": "flat", "fee_bps_per_side": 6.0, "slip_bps_per_side": 9.0,
          "funding_default_8h": 0.0001}


def _with_costs(cfg: StrategyConfig, costs: dict) -> StrategyConfig:
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
            t["R_gross"] = t["gross_pnl"] / t["risk"]
            t["edge_pct"] = t["gross_pnl"] / t["notional"]
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    barriers = {}
    for tf in ("1h", "4h", "1d"):
        try:
            with open(f"lab_recost_{tf}.json", encoding="utf-8") as f:
                for r in json.load(f)["results"]:
                    barriers[r["name"]] = (r["barriers"], r["robustness"])
        except FileNotFoundError:
            pass

    rows = []
    for tf in ("1h", "4h", "1d"):
        ds = load_basket("./data", tf=tf)
        pool = [(h, "мейкер (лимит)") for h, _ in recost_pool(tf)]
        # рыночные версии fade — для контраста: у них вход тейкерный, и
        # уточнение тарифа им почти ничего не даёт
        if tf in ("1d", "4h"):
            pool += [(h, "рынок (тейкер)") for h, _ in fade_pool(tf)]
        for hypo, kind in pool:
            cfg_new = hypo.config
            if cfg_new.is_flat():
                cfg_new = _with_costs(cfg_new, dict(
                    __import__("crypto_strat.search.recost", fromlist=["x"]).REAL_COSTS))
            cfg_old = _with_costs(cfg_new, LEGACY)

            t_new = _run(cfg_new, ds)
            t_old = _run(cfg_old, ds)
            if not len(t_new):
                continue

            # стресс: удвоенное ПРОСКАЛЬЗЫВАНИЕ (комиссия остаётся реальной)
            d = cfg_new.to_dict()
            d["costs"]["slip_bps"] = float(d["costs"]["slip_bps"]) * 2.0
            t_stress = _run(StrategyConfig.from_dict(d), ds)

            years = (t_new.assign(y=t_new.entry_dt.dt.year).groupby("y")["R"]
                     .agg(["size", "mean"]))
            rows.append({
                "name": hypo.name, "tf": tf, "kind": kind,
                "n_new": len(t_new), "n_old": len(t_old),
                "gross_R": float(t_new["R_gross"].mean()),
                "gross_pct": float(t_new["edge_pct"].mean()),
                "rt_bps": float(10_000 * t_new["fees"].sum() / t_new["notional"].sum()),
                "rt_bps_old": float(10_000 * t_old["fees"].sum() / t_old["notional"].sum())
                              if len(t_old) else float("nan"),
                "net_old": float(t_old["R"].mean()) if len(t_old) else float("nan"),
                "net_new": float(t_new["R"].mean()),
                "net_stress": float(t_stress["R"].mean()) if len(t_stress) else float("nan"),
                "stop_atr_pct": float(t_new["stop_dist_pct"].median()),
                "fresh24": float(t_new.loc[t_new.entry_dt >= FRESH, "R"].mean()),
                "fresh25": float(t_new.loc[t_new.entry_dt >= RECENT, "R"].mean()),
                "old_era": float(t_new.loc[t_new.entry_dt < SPLIT, "R"].mean()),
                "new_era": float(t_new.loc[t_new.entry_dt >= SPLIT, "R"].mean()),
                "barriers": barriers.get(hypo.name, (None, None))[0],
                "years": {int(y): {"n": int(r["size"]), "net": float(r["mean"])}
                          for y, r in years.iterrows()},
            })
            print(f"  посчитано: {hypo.name}")

    print("\n" + "=" * 132)
    print("ПЕРЕРАСЧЁТ НА РЕАЛЬНЫХ ИЗДЕРЖКАХ (Bybit maker 0.02% / taker 0.055%, "
          "проскальзывание 9 bps НЕ менялось)")
    print("=" * 132)
    print(f"{'кандидат':<26}{'ТФ':<4}{'вход':<16}{'сделок':>7}{'валовой,R':>11}"
          f"{'вал,%нот':>10}{'rt bps':>8}{'нетто СТАР':>11}{'нетто НОВ':>11}"
          f"{'слип x2':>9}{'барьеров':>10}")
    for r in sorted(rows, key=lambda x: -(x["barriers"] or 0)):
        b = f"{r['barriers']}/7" if r["barriers"] is not None else "—"
        print(f"{r['name'][:25]:<26}{r['tf']:<4}{r['kind']:<16}{r['n_new']:>7}"
              f"{r['gross_R']:>+11.3f}{r['gross_pct']*100:>+9.3f}%{r['rt_bps']:>8.1f}"
              f"{r['net_old']:>+11.3f}{r['net_new']:>+11.3f}{r['net_stress']:>+9.3f}"
              f"{b:>10}")

    print("\n" + "-" * 132)
    print("СВЕЖЕСТЬ ЭДЖА (нетто на реальных издержках) — обязательное условие")
    print(f"{'кандидат':<26}{'ТФ':<4}{'вся история':>13}{'2017-21':>10}"
          f"{'2022-26':>10}{'с 2024':>10}{'с 2025':>10}  вердикт")
    for r in sorted(rows, key=lambda x: -x["net_new"]):
        ok = r["net_new"] > 0 and r["fresh24"] > 0 and r["fresh25"] > 0
        v = ("свежесть ОК" if ok else
             ("historical-only" if r["net_new"] > 0 else "нетто в минусе"))
        print(f"{r['name'][:25]:<26}{r['tf']:<4}{r['net_new']:>+13.3f}"
              f"{r['old_era']:>+10.3f}{r['new_era']:>+10.3f}"
              f"{r['fresh24']:>+10.3f}{r['fresh25']:>+10.3f}  {v}")

    pos = [r for r in rows if r["net_new"] > 0]
    print("\n" + "-" * 132)
    print(f"вышли в положительный нетто при реальных издержках: {len(pos)} из {len(rows)}")
    for r in sorted(pos, key=lambda x: -x["net_new"]):
        print(f"\n  {r['name']} ({r['tf']}): нетто {r['net_old']:+.3f} -> "
              f"{r['net_new']:+.3f}R, сделок {r['n_new']}, "
              f"барьеров {r['barriers']}/7")
        print("    нетто по годам: " + "  ".join(
            f"{y}:{d['net']:+.2f}({d['n']})" for y, d in sorted(r["years"].items())))
        print(f"    свежесть: с 2024 {r['fresh24']:+.3f}R | с 2025 {r['fresh25']:+.3f}R"
              f" | при слиппедже x2 {r['net_stress']:+.3f}R")

    with open("recost_analysis.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\nОтчёт: recost_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
