"""
Отдельный прогон order block трейдера через фильтр (по запросу).

Почему отдельным скриптом, а не в общей очереди: по р.6.5 собственные сетапы
трейдера — калибровочный эталон, а не сырьё для поиска Фазы 1. Здесь они
проходят фильтр НА ОБЩИХ ОСНОВАНИЯХ, без поблажек за происхождение, но
запускаются осознанно и штучно.

Счётчик проб НЕ обнуляется: прогон идёт по тем же данным, что и поиск,
значит смещение отбора переносится и планка барьера 5 обязана быть выше.

Запуск:
    python run_order_block.py --data ./data --tf 4h
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from calibrate_filter import honest_order_block
from crypto_strat.data.loader import load_basket
from crypto_strat.validation.barriers import Thresholds
from crypto_strat.validation.filter import run_filter
from crypto_strat.validation.hypothesis import TrialLog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--primary", default=None)
    ap.add_argument("--trials", default="./trials_search.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset = load_basket(args.data, tf=args.tf)
    if not dataset:
        print(f"Нет CSV для ТФ {args.tf} в {args.data}")
        return 3

    primary = [args.primary] if args.primary else [list(dataset)[0]]
    hypo = honest_order_block()
    hypo.config.timeframe = args.tf
    trial_log = TrialLog(args.trials)

    print("=" * 78)
    print(f"ORDER BLOCK ТРЕЙДЕРА — ФИЛЬТР, ТФ {args.tf}")
    print("=" * 78)
    print(f"Инструменты: {', '.join(dataset)} | подбор параметров на {primary[0]}")
    print(f"Комбинаций в сетке: {len(hypo.variants())}")
    print(f"Конфигов перебрано ранее по этим данным: {trial_log.total} "
          f"(входит в планку барьера 5)")
    print("=" * 78)

    v = run_filter(dataset, hypo, primary=primary, th=Thresholds(),
                   trial_log=trial_log, fail_fast=False)
    print(v.report())

    out = args.out or f"order_block_{args.tf}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "tf": args.tf, "primary": primary, "symbols": list(dataset),
            "survived": v.survived, "barriers_passed": v.n_passed,
            "robustness": round(v.robustness, 4),
            "failed_barriers": [b.name for b in v.barriers if not b.passed],
            "barrier_notes": {b.name: b.note for b in v.barriers},
            "barrier_detail": {b.name: b.detail for b in v.barriers},
            "best_config": v.best_config.to_dict() if v.best_config else None,
            "metrics": v.metrics, "flags": v.flags,
            "trials_total": trial_log.total,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОтчёт: {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
