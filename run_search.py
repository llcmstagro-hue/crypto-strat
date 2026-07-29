"""
Этап 5 — прогон очереди гипотез через фильтр анти-оверфита.

Два режима, и разница между ними принципиальна:

  --data ./data   РЕАЛЬНЫЕ CSV. Результат — настоящий: выжившие считаются
                  кандидатами на форвард, «никто не выжил» считается валидным
                  ответом (р.7) и не является поводом ослабить пороги.

  без --data      СИНТЕТИКА. Это ТОЛЬКО проверка проводки: что все блоки
                  собираются, все барьеры отрабатывают, ничего не падает.
                  Метрики отсюда НЕ являются доказательством эджа, и скрипт
                  явно отказывается называть кого-либо выжившим кандидатом.

Число перебранных конфигов копится в trials_search.json и подаётся в барьер 5:
чем шире был перебор, тем выше планка победителю (р.4, п.5).

Запуск:
    python run_search.py --data ./data --tf 1h
    python run_search.py --smoke            # проверка проводки на синтетике
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from crypto_strat.data.loader import load_basket
from crypto_strat.data.synthetic import make_basket
from crypto_strat.search.registry import phase1_queue, dedup
from crypto_strat.validation.barriers import Thresholds
from crypto_strat.validation.filter import run_batch, summary_table
from crypto_strat.validation.hypothesis import TrialLog

SMOKE_NOTE = ("СИНТЕТИКА: прогон проверяет проводку конвейера, а не гипотезы. "
              "Никакой результат отсюда не является доказательством эджа.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="папка с реальными CSV")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--primary", default=None, help="инструмент для подбора параметров")
    ap.add_argument("--smoke", action="store_true", help="прогон на синтетике")
    ap.add_argument("--bars", type=int, default=35_000)
    ap.add_argument("--full-report", action="store_true",
                    help="прогонять все барьеры даже после первого провала")
    ap.add_argument("--out", default="search_results.json")
    ap.add_argument("--trials", default="./trials_search.json")
    args = ap.parse_args()

    real = False
    if args.data and os.path.isdir(args.data):
        dataset = load_basket(args.data, tf=args.tf)
        real = bool(dataset)
        if not real:
            print(f"! в {args.data} нет подходящих CSV для ТФ {args.tf}")
    else:
        dataset = {}

    if not real:
        if not args.smoke:
            print("Реальных данных нет. Запуск поиска на синтетике бессмысленен: "
                  "он даст цифры, ничего не значащие про рынок.\n"
                  "Если нужна проверка проводки конвейера — запусти с --smoke.\n"
                  "Реальные данные: python download_data.py --source bulk "
                  "--tf 1h 4h 1d --start 2022-01 --funding")
            return 3
        dataset = make_basket(regime="edge", n=args.bars, tf=args.tf)

    primary = [args.primary] if args.primary else [list(dataset)[0]]
    hypos = dedup(phase1_queue())
    trial_log = TrialLog(args.trials)

    print("=" * 92)
    print("ЭТАП 5 — ПРОГОН ГИПОТЕЗ ЧЕРЕЗ ФИЛЬТР")
    print("=" * 92)
    print(f"Данные:    {'РЕАЛЬНЫЕ (' + args.data + ')' if real else 'СИНТЕТИКА (smoke)'}")
    print(f"ТФ:        {args.tf} | инструментов: {len(dataset)} "
          f"({', '.join(dataset)})")
    print(f"Подбор параметров только на: {primary[0]}")
    print(f"Гипотез в очереди: {len(hypos)}")
    print(f"Конфигов перебрано ранее (накопленный счётчик): {trial_log.total}")
    if not real:
        print(f"⚠️  {SMOKE_NOTE}")
    print("=" * 92)

    verdicts = run_batch(dataset, hypos, primary=primary, th=Thresholds(),
                         trial_log=trial_log, fail_fast=not args.full_report,
                         data_note="" if real else SMOKE_NOTE)

    for v in verdicts:
        print(v.report())
    print(summary_table(verdicts, trial_log))

    survivors = [v for v in verdicts if v.survived]
    if not real:
        print("\n⚠️  Прогон синтетический. Ни одна из строк выше НЕ является "
              "кандидатом на форвард — это проверка проводки.")
    elif survivors:
        print("\nВЫЖИВШИЕ (ранжированы ПО РОБАСТНОСТИ, не по доходности):")
        for i, v in enumerate(survivors, 1):
            m = v.metrics
            print(f"  {i}. {v.hypothesis}  робастность={v.robustness:.3f}")
            print(f"     конфиг: {v.best_config.name}")
            print(f"     честные метрики: сделок {m['n_trades']}, "
                  f"winrate {m['winrate']*100:.1f}%, PF {m['profit_factor']:.2f}, "
                  f"exp {m['expectancy_R']:+.3f}R, макс. серия убытков "
                  f"{m['max_loss_streak']}")
            print(f"     источник: {v.source}")
    else:
        print("\nЭДЖА НЕ НАЙДЕНО. В проверенном пространстве гипотез ни одна не "
              "пережила все барьеры.\nЭто валидный результат (р.7). Пороги не "
              "трогаем, метрики не подкручиваем.")

    payload = {
        "real_data": real, "tf": args.tf, "primary": primary,
        "symbols": list(dataset),
        "trials_total": trial_log.total, "hypotheses_run": trial_log.hypotheses,
        "results": [{
            "hypothesis": v.hypothesis, "source": v.source,
            "survived": v.survived, "barriers_passed": v.n_passed,
            "robustness": round(v.robustness, 4),
            "failed_barriers": [b.name for b in v.barriers if not b.passed],
            "barrier_notes": {b.name: b.note for b in v.barriers},
            "best_config": v.best_config.to_dict() if v.best_config else None,
            "metrics": {k: v.metrics.get(k) for k in
                        ("n_trades", "winrate", "profit_factor", "expectancy_R",
                         "sharpe_trade", "total_R", "max_dd_R", "max_loss_streak")},
            "flags": v.flags,
        } for v in verdicts],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОтчёт: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
