"""
Бумажный форвард — заморозка стратегии и наблюдение (модуль 6, р.6.6).

Стратегия здесь НЕ получает боевого допуска. Она не прошла фильтр, и это не
изменилось. Наблюдение отвечает на единственный вопрос, на который бэктест
ответить не может: **эдж умер или эдж спит?**

Логика простая. Бэктест смотрит в прошлое, где эдж БЫЛ, и потому не может
отличить «затух насовсем» от «пережидает неудобный режим». Живые данные,
которых стратегия не видела ни в каком виде, могут.

    python run_forward.py freeze --data ./data --tf 4h \\
        --name donchian_breakout+filter_htf
    python run_forward.py status --data ./data       # когда появятся новые бары

Параметры после заморозки НЕ МЕНЯЮТСЯ. Никогда. Подкрутить их по ходу
наблюдения = превратить наблюдение в подгонку в динамике (р.6.6).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

from crypto_strat.data.loader import load_basket
from crypto_strat.engine.config import StrategyConfig
from crypto_strat.engine.metrics import trade_metrics
from crypto_strat.forward.monitor import (FrozenStrategy, evaluate, format_status,
                                          monte_carlo_baseline, thresholds_at)
from crypto_strat.validation.barriers import run_symbols, pooled_R

STORE = "./forward"


def _load_config_from_kb(db: str, name: str, tf: str) -> tuple[dict, dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT config_json, metrics, reason, barriers_passed FROM hypotheses "
        "WHERE name=? AND tf=?", (name, tf)).fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"в базе знаний нет {name} на {tf}")
    return json.loads(row["config_json"]), {
        "reason": row["reason"], "barriers_passed": row["barriers_passed"]}


def cmd_freeze(args) -> int:
    os.makedirs(STORE, exist_ok=True)
    cfg_dict, kb = _load_config_from_kb(args.db, args.name, args.tf)
    cfg = StrategyConfig.from_dict(cfg_dict)
    dataset = load_basket(args.data, tf=args.tf)
    if not dataset:
        raise SystemExit(f"нет данных {args.tf} в {args.data}")

    results = run_symbols(dataset, cfg, list(dataset))
    R = pooled_R(results)
    base = trade_metrics(R)
    mc = monte_carlo_baseline(R, horizon=min(len(R), 200))

    data_end = max(str(d["ohlcv"]["dt_utc"].iloc[-1].date()) for d in dataset.values())
    frozen = FrozenStrategy(
        name=args.name, config=cfg.to_dict(), tf=args.tf,
        symbols=list(dataset), frozen_data_end=data_end,
        why_watching=(
            f"прошла {kb['barriers_passed']}/7 барьеров и проверку по режимам, "
            f"но эдж измеримо затух (p<0.001) и отрицателен в 2026. "
            f"Наблюдаем, чтобы отличить «эдж умер» от «эдж спит»."),
        baseline={k: base[k] for k in
                  ("n_trades", "winrate", "profit_factor", "expectancy_R",
                   "sharpe_trade", "max_dd_R", "max_loss_streak")},
        monte_carlo=mc)

    path = os.path.join(STORE, f"{args.name.replace('+', '_')}_{args.tf}.json")
    frozen.save(path)
    print(format_status(frozen, evaluate(frozen, None)))
    print(f"\nЗаморожено: {os.path.abspath(path)}")
    print("Параметры больше не меняются. Запускай `status` по мере появления данных.")
    return 0


def cmd_status(args) -> int:
    files = ([args.file] if args.file else
             [os.path.join(STORE, f) for f in sorted(os.listdir(STORE))
              if f.endswith(".json")] if os.path.isdir(STORE) else [])
    if not files:
        print("Нечего наблюдать — сначала `freeze`.")
        return 1

    for path in files:
        frozen = FrozenStrategy.load(path)
        dataset = load_basket(args.data, tf=frozen.tf,
                              symbols=frozen.symbols)
        if not dataset:
            print(f"! нет данных {frozen.tf} для {frozen.name}")
            continue

        cfg = StrategyConfig.from_dict(frozen.config)
        results = run_symbols(dataset, cfg, list(dataset))
        cutoff = pd.Timestamp(frozen.frozen_data_end, tz="UTC").value // 10**6

        # только сделки, ОТКРЫТЫЕ после точки заморозки
        fresh = [r.trades[r.trades["entry_ts"] > cutoff] for r in results if r.n_trades]
        fresh = [t for t in fresh if len(t)]
        trades = pd.concat(fresh).sort_values("entry_ts") if fresh else None

        ev = evaluate(frozen, trades)
        print(format_status(frozen, ev))
        if trades is not None and len(trades):
            print("\n  Последние сделки наблюдения:")
            cols = ["entry_dt", "exit_dt", "direction", "R", "exit_reason"]
            print(trades.tail(10)[[c for c in cols if c in trades.columns]]
                  .to_string(index=False))
    return 0


def cmd_replay(args) -> int:
    """Проверка САМОГО МОНИТОРА на истории.

    Замораживаем стратегию задним числом на дату `--as-of`, считаем baseline и
    пороги Монте-Карло ТОЛЬКО по данным до неё, затем «наблюдаем» всё, что было
    после. Это бэктест МОНИТОРА, а не стратегии: он отвечает на вопрос
    «сработали бы жёлтый и красный триггеры там, где эдж действительно затухал».

    Без такой проверки монитор — необоснованный код: пороги посчитаны, но
    неизвестно, ловят ли они то, ради чего заведены.
    """
    cfg_dict, kb = _load_config_from_kb(args.db, args.name, args.tf)
    cfg = StrategyConfig.from_dict(cfg_dict)
    dataset = load_basket(args.data, tf=args.tf)
    cut = pd.Timestamp(args.as_of, tz="UTC").value // 10**6

    results = run_symbols(dataset, cfg, list(dataset))
    allt = pd.concat([r.trades for r in results if r.n_trades]).sort_values("entry_ts")
    before = allt[allt.entry_ts < cut]
    after = allt[allt.entry_ts >= cut]
    if len(before) < 50 or len(after) < 10:
        raise SystemExit(f"мало сделок для реплея: до {len(before)}, после {len(after)}")

    base = trade_metrics(before["R"].to_numpy())
    mc = monte_carlo_baseline(before["R"].to_numpy(), horizon=min(len(after), 300))
    frozen = FrozenStrategy(
        name=f"{args.name} [РЕПЛЕЙ as-of {args.as_of}]", config=cfg.to_dict(),
        tf=args.tf, symbols=list(dataset), frozen_data_end=args.as_of,
        why_watching="проверка монитора на истории: поймал бы он затухание?",
        baseline={k: base[k] for k in ("n_trades", "winrate", "profit_factor",
                                       "expectancy_R", "sharpe_trade",
                                       "max_dd_R", "max_loss_streak")},
        monte_carlo=mc)

    print(format_status(frozen, evaluate(frozen, after)))

    # когда именно загорелся бы каждый уровень
    R = after["R"].to_numpy()
    eq = np.cumsum(R)
    dd = np.maximum.accumulate(eq) - eq
    ts = after["entry_dt"].to_numpy()
    # Порог на сделке #k берётся ДЛЯ k сделок, а не фиксированный: иначе
    # ранние сделки судятся по слишком мягкой планке, поздние — по слишком
    # жёсткой, и момент срабатывания получается фиктивным.
    print("\n  КОГДА СРАБОТАЛИ БЫ ТРИГГЕРЫ (плоский порог на весь горизонт):")
    ty, tr = thresholds_at(mc, len(R))
    thr_y = np.full(len(R), ty); thr_r = np.full(len(R), tr)
    for lvl, thr in (("🟡 жёлтый", thr_y), ("🔴 красный", thr_r)):
        hit = int(np.argmax(dd >= thr)) if (dd >= thr).any() else None
        if hit is None:
            print(f"     {lvl}: не сработал за весь период наблюдения")
        else:
            print(f"     {lvl}: сделка #{hit+1} из {len(R)}, "
                  f"{pd.Timestamp(ts[hit]).date()} "
                  f"(просадка {dd[hit]:.1f}R при пороге {thr[hit]:.1f}R)")

    # катящееся ожидание — против КАЛИБРОВАННОГО порога, а не против нуля
    rl = mc.get("rolling", {})
    w = mc.get("rolling_n", 40)
    if len(R) >= w and rl:
        roll = pd.Series(R).rolling(w).mean()
        for lvl, q in (("🟡 жёлтый", rl["p05_any_window"]),
                       ("🔴 красный", rl["p01_worst_window"])):
            below = roll[roll <= q]
            if len(below):
                i = int(below.index[0])
                print(f"     {lvl} по ожиданию ({q:+.3f}R): сделка #{i+1}, "
                      f"{pd.Timestamp(ts[i]).date()}")
            else:
                print(f"     {lvl} по ожиданию ({q:+.3f}R): не сработал")
        naive = roll[roll < 0]
        print(f"     справочно: наивный триггер «ожидание < 0» сработал бы "
              f"{int(((roll < 0).astype(int).diff() == 1).sum())} раз "
              f"за {len(R)} сделок")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="заморозить стратегию под наблюдение")
    f.add_argument("--data", default="./data")
    f.add_argument("--tf", default="4h")
    f.add_argument("--db", default="./knowledge.db")
    f.add_argument("--name", required=True)
    f.set_defaults(func=cmd_freeze)

    s = sub.add_parser("status", help="как ведёт себя замороженная стратегия")
    s.add_argument("--data", default="./data")
    s.add_argument("--file", default=None)
    s.set_defaults(func=cmd_status)

    rp = sub.add_parser("replay", help="проверить монитор на истории")
    rp.add_argument("--data", default="./data")
    rp.add_argument("--tf", default="4h")
    rp.add_argument("--db", default="./knowledge.db")
    rp.add_argument("--name", required=True)
    rp.add_argument("--as-of", required=True, help="дата ретроспективной заморозки")
    rp.set_defaults(func=cmd_replay)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
