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

import pandas as pd

from crypto_strat.data.loader import load_basket
from crypto_strat.engine.config import StrategyConfig
from crypto_strat.engine.metrics import trade_metrics
from crypto_strat.forward.monitor import (FrozenStrategy, evaluate, format_status,
                                          monte_carlo_baseline)
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

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
