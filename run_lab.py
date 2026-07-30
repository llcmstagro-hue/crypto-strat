"""
Trading Research Lab — широкий честный поиск (р.6.7).

Конвейер на каждую гипотезу:

    дедуп по базе знаний
        -> Research Score + Critic   (ДЕШЁВЫЙ гейт, до бэктеста)
        -> фильтр анти-оверфита р.4  (ВСЕ 7 барьеров)
        -> проверка по режимам рынка (медвежка И бык, не только рост)
        -> запись в базу знаний с вердиктом и причиной

«Достойная стратегия» = прошла ВСЕ 7 барьеров И держит эдж в разных режимах.
Одного 7/7 недостаточно: стратегия, сделавшая всю прибыль в бычьем 2021,
на челлендже встретит медвежий период и выбьет лимит.

Что здесь СОЗНАТЕЛЬНО не делается:
  * не выбирается «лучший вариант по истории» (это ядовитая версия Evolution);
  * не снижаются пороги, если никто не проходит;
  * не оптимизируется доходность — только робастность.

Запуск:
    python run_lab.py --data ./data --tf 4h
    python run_lab.py --data ./data --tf 4h --no-evolution      # только база
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from crypto_strat.data.loader import load_basket
from crypto_strat.knowledge.db import KnowledgeBase, Record
from crypto_strat.search.evolution import evolve
from crypto_strat.search.ensembles import (ensemble_pool, ensemble_seeds,
                                          conqueror_pool, MAX_HYPOTHESES)
from crypto_strat.search.pool import base_pool, evolution_seeds, new_classes_pool
from crypto_strat.search.smith import smith_pool
from crypto_strat.search.fade import fade_pool
from crypto_strat.search.score import Idea, classify, gate
from crypto_strat.validation.barriers import Thresholds, run_symbols, thresholds_for_tf
from crypto_strat.validation.filter import run_filter
from crypto_strat.validation.hypothesis import TrialLog
from crypto_strat.validation.regimes import (regime_breakdown, regime_verdict,
                                             format_regimes, freshness_check)


EVO_SEEDS_NEW = ("vol_squeeze_breakout", "vol_ttm_squeeze", "vol_nr_expansion",
                 "xmkt_btc_spillover", "xmkt_btc_seesaw", "xmkt_donchian_btc_filter")


def build_pool(with_evolution: bool = True, classes: str = "all", tf: str = "4h"):
    """classes: all | new | base.

    `new` — только волатильностный и межрыночный классы. Пробой и возврат к
    среднему изучены и закрыты (см. PROGRESS.md): гонять их снова значит
    поднимать планку барьера 5 всем новым кандидатам без шанса узнать что-то
    новое.
    """
    OLD_SEEDS = ("donchian_breakout", "keltner_breakout", "tsmom",
                 "bb_meanrev", "order_block", "level_retest")
    if classes == "conqueror":
        # внешняя стратегия проверяется КАК ЕСТЬ: без Evolution, без мутаций.
        # Задача — вердикт по чужому методу, а не поиск удачной его версии.
        return conqueror_pool(tf)
    if classes == "fade":
        # веб-гипотеза проверяется как есть: без Evolution. Вопрос прогона —
        # «ведёт ли себя fade иначе», а не «найдётся ли удачная его версия».
        return fade_pool(tf)
    if classes == "smith":
        # то же правило, что и для Conqueror: чужие методы гоняются как есть.
        # Мутировать их — значит проверять уже не Смита, а нас.
        return smith_pool(tf)
    if classes == "ensemble":
        pool, seed_names = ensemble_pool(), ensemble_seeds()
    elif classes == "new":
        pool, seed_names = new_classes_pool(), EVO_SEEDS_NEW
    elif classes == "base":
        pool, seed_names = base_pool(), OLD_SEEDS
    else:
        pool = base_pool() + new_classes_pool()
        seed_names = OLD_SEEDS + EVO_SEEDS_NEW

    if not with_evolution:
        return pool
    seeds = [(h, i) for h, i in pool if h.name in (seed_names or ())]
    grown = list(pool)
    for h, idea in seeds:
        grown.extend(evolve(h.config, idea, h.grid,
                            primary=h.primary, exclude_symbols=h.exclude_symbols))
    # ЛИМИТ ШИРИНЫ: каждая лишняя гипотеза поднимает планку барьера 5 всем
    # будущим кандидатам. Обрезаем детерминированно и говорим об этом вслух,
    # а не молча — молчаливое усечение читается как «проверили всё».
    if len(grown) > MAX_HYPOTHESES:
        print(f"[лимит ширины] пул {len(grown)} -> {MAX_HYPOTHESES} гипотез; "
              f"отброшено {len(grown) - MAX_HYPOTHESES} вариантов Evolution "
              f"(базовые гипотезы сохранены полностью)")
        base_n = len(pool)
        grown = pool + grown[base_n:base_n + max(0, MAX_HYPOTHESES - base_n)]
    return grown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--primary", default="BTCUSDT")
    ap.add_argument("--db", default="./knowledge.db")
    ap.add_argument("--trials", default="./trials_lab.json")
    ap.add_argument("--no-evolution", action="store_true")
    ap.add_argument("--classes", default="all",
                    choices=["all", "new", "base", "ensemble", "conqueror",
                             "smith", "fade"],
                    help="какие классы механизмов гонять")
    ap.add_argument("--fail-fast", action="store_true",
                    help="останавливать гипотезу на первом проваленном барьере")
    ap.add_argument("--out", default="lab_results.json")
    args = ap.parse_args()

    dataset = load_basket(args.data, tf=args.tf)
    if not dataset:
        print(f"Нет данных для ТФ {args.tf} в {args.data}")
        return 3
    has_vol = bool(next(iter(dataset.values()))["ohlcv"].attrs.get("has_volume", True))
    spans = {s: (str(d["ohlcv"]["dt_utc"].iloc[0].date()),
                 str(d["ohlcv"]["dt_utc"].iloc[-1].date()), len(d["ohlcv"]))
             for s, d in dataset.items()}
    data_span = f"{min(v[0] for v in spans.values())}..{max(v[1] for v in spans.values())}"

    kb = KnowledgeBase(args.db)
    tl = TrialLog(args.trials)
    th = thresholds_for_tf(args.tf)
    pool = build_pool(not args.no_evolution, args.classes, args.tf)

    print("=" * 96)
    print("TRADING RESEARCH LAB — ШИРОКИЙ ПОИСК")
    print("=" * 96)
    for s, (a, b, n) in spans.items():
        print(f"  {s:<10} {n:>6} баров  {a} .. {b}")
    print(f"  объём в данных: {'есть' if has_vol else 'НЕТ (объёмные блоки исключены)'}")
    print(f"  ТФ {args.tf} | подбор параметров только на {args.primary}")
    print(f"  гипотез в пуле: {len(pool)}  (классы: {args.classes} + Evolution)")
    print(f"  база знаний: {os.path.abspath(args.db)}")
    print(f"  порог Research Score: 70 | DSR: {th.dsr_min} | мин. сделок: "
          f"{th.min_trades_total}/{th.min_trades_per_symbol}")
    print(f"  walk-forward: train {th.wf_train_days}д / test {th.wf_test_days}д "
          f"(длина окна подобрана под плотность ТФ; критерии прохождения те же)")
    print("=" * 96)

    t_start = time.time()
    worthy, results = [], []

    for i, (hypo, idea) in enumerate(pool, 1):
        fp = hypo.config.fingerprint()
        prev = kb.seen(fp, args.tf)
        is_dup = prev is not None
        stype = classify(hypo.config)

        rec = Record(fingerprint=fp, name=hypo.name, tf=args.tf, source=hypo.source,
                     strategy_type=stype, generation=("evolution" if "+" in hypo.name
                                                      else "base"),
                     config=hypo.config.to_dict(), grid=hypo.grid,
                     rationale=idea.rationale, data_span=data_span)

        g = gate(hypo.config, hypo.grid, idea, has_volume=has_vol, is_duplicate=is_dup)
        rec.research_score = g["score"]
        rec.score_breakdown = g["breakdown"]
        rec.critic_verdict = g["critic"]
        rec.critic_objections = g["objections"]

        head = f"[{i}/{len(pool)}] {hypo.name}  ({stype}, score {g['score']})"
        if not g["admitted"]:
            rec.tested = False
            rec.reason = f"не допущена к бэктесту: {g['reason']}"
            kb.save(rec)
            print(f"\n{head}\n    ⛔ {rec.reason}")
            continue

        print(f"\n{head}")
        if g["objections"]:
            print(f"    Critic: {'; '.join(g['objections'])}")

        # межрыночные гипотезы работают на суженной корзине и своём primary
        sub = {k: v_ for k, v_ in dataset.items() if k not in hypo.exclude_symbols}
        prim = hypo.primary or args.primary
        if prim not in sub:
            prim = list(sub)[0]
        if hypo.exclude_symbols:
            print(f"    корзина: {', '.join(sub)} | подбор на {prim} "
                  f"(ведущий инструмент исключён из ведомых)")

        v = run_filter(sub, hypo, primary=[prim], th=th,
                       trial_log=tl, fail_fast=args.fail_fast, verbose=True)

        rec.tested = True
        rec.barriers_passed = v.n_passed
        rec.failed_barriers = [b.name for b in v.barriers if not b.passed]
        rec.robustness = round(v.robustness, 4)
        rec.metrics = {k: v.metrics.get(k) for k in
                       ("n_trades", "winrate", "profit_factor", "expectancy_R",
                        "sharpe_trade", "total_R", "max_dd_R", "max_loss_streak",
                        "profit_concentration")}
        rec.trials_at_test = tl.selection

        # --- режимы и СВЕЖЕСТЬ: считаем всегда, когда есть фиксированный конфиг ---
        reg_pass, reg_note = False, "не считались"
        fresh_pass, fresh_note = False, "не считалась"
        if v.best_config is not None:
            res = run_symbols(sub, v.best_config, list(sub))
            bd = regime_breakdown(res)
            rv = regime_verdict(bd)
            fr = freshness_check(res)
            rec.regimes = {"breakdown": bd["rows"], "verdict": rv, "freshness": fr}
            reg_pass, reg_note = rv["passed"], rv["note"]
            fresh_pass, fresh_note = fr["passed"], fr["note"]
            if v.survived or v.n_passed >= 5:
                print("    режимы:")
                for line in format_regimes(bd).splitlines():
                    print("      " + line)
                print(f"    {'✅' if reg_pass else '❌'} режимы: {reg_note}")
                print(f"    {'✅' if fresh_pass else '❌'} свежесть: {fresh_note}")

        is_worthy = v.survived and reg_pass and fresh_pass
        rec.survived = bool(is_worthy)
        if is_worthy:
            rec.reason = ("прошла все 7 барьеров, держит эдж в разных режимах "
                          "И эдж жив сейчас")
            worthy.append((hypo, v, rec))
        elif v.survived and reg_pass and not fresh_pass:
            # ровно тот случай, ради которого критерий и введён
            rec.reason = f"historical-only edge: 7/7 и режимы ОК, но эдж мёртв сейчас — {fresh_note}"
        elif v.survived:
            rec.reason = f"7/7 барьеров, но НЕ прошла по режимам: {reg_note}"
        else:
            rec.reason = f"провалила барьеры: {', '.join(rec.failed_barriers)}"
        kb.save(rec)

        for fl in v.flags:
            print(f"    🚩 {fl}")
        print(f"    ИТОГ: {'ДОСТОЙНАЯ' if is_worthy else ('7/7 но режимы' if v.survived else 'отсев')}"
              f" ({v.n_passed}/7, робастность {v.robustness:.3f}"
              + (", эдж мёртв сейчас" if (v.survived and reg_pass and not fresh_pass) else "")
              + ")")
        results.append({"name": hypo.name, "type": stype, "score": g["score"],
                        "barriers": v.n_passed, "robustness": rec.robustness,
                        "regimes_ok": reg_pass, "fresh_ok": fresh_pass,
                        "worthy": is_worthy,
                        "metrics": rec.metrics,
                        "failed": rec.failed_barriers,
                        "config": v.best_config.to_dict() if v.best_config else None})

    # ------------------------------------------------------------------ #
    el = time.time() - t_start
    print("\n" + "=" * 96)
    print("ИТОГИ ПРОГОНА")
    print("=" * 96)
    st = kb.stats()
    for k, val in st.items():
        print(f"  {k}: {val}")
    print(f"  конфигов перебрано: {tl.total} всего / {tl.selection} в пуле отбора "
          f"(пул отбора — вход барьера 5)")
    print(f"  время: {el/60:.1f} мин")

    print("\n  Какие ТИПЫ идей выживают (накопленное знание):")
    print(f"    {'тип':<18}{'всего':>7}{'тест':>7}{'выжило':>8}{'ср.барьеров':>13}{'ср.робастн':>12}")
    for r in kb.by_type():
        print(f"    {str(r['тип']):<18}{r['всего']:>7}{r['тестировано'] or 0:>7}"
              f"{r['выжило'] or 0:>8}{(r['ср_барьеров'] or 0):>13.2f}"
              f"{(r['ср_робастность'] or 0):>12.3f}")

    print("\n  Где чаще всего умирают:")
    for name, n in kb.failure_reasons():
        print(f"    {n:>4}x  {name}")

    ranked = sorted(results, key=lambda r: (r["worthy"], r["barriers"], r["robustness"]),
                    reverse=True)
    print("\n" + "-" * 96)
    print(f"{'гипотеза':<38}{'тип':<16}{'барьеров':>9}{'робастн':>9}{'сделок':>8}{'exp,R':>9}  итог")
    for r in ranked[:30]:
        m = r["metrics"] or {}
        print(f"{r['name'][:37]:<38}{r['type']:<16}{r['barriers']}/7{'':>5}"
              f"{r['robustness']:>9.3f}{m.get('n_trades') or 0:>8}"
              f"{(m.get('expectancy_R') or 0):>+9.3f}  "
              f"{'ДОСТОЙНАЯ' if r['worthy'] else ('7/7-режимы' if r['barriers']==7 else 'отсев')}")

    print("\n" + "=" * 96)
    if worthy:
        print(f"ДОСТОЙНЫЕ СТРАТЕГИИ: {len(worthy)}")
        for hypo, v, rec in sorted(worthy, key=lambda x: -x[1].robustness):
            m = rec.metrics
            print(f"\n  ★ {hypo.name}  робастность {v.robustness:.3f}")
            print(f"    конфиг: {v.best_config.name}")
            print(f"    сделок {m['n_trades']}, winrate {m['winrate']*100:.1f}%, "
                  f"PF {m['profit_factor']:.2f}, exp {m['expectancy_R']:+.3f}R, "
                  f"макс. серия убытков {m['max_loss_streak']}")
            print(f"    источник: {hypo.source}")
    else:
        print("ДОСТОЙНЫХ СТРАТЕГИЙ НЕ НАЙДЕНО.")
        print("Это валидный результат (р.7). Пороги не трогались, метрики не "
              "подкручивались.")
    print("=" * 96)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tf": args.tf, "primary": args.primary, "data_span": data_span,
                   "spans": spans, "has_volume": has_vol,
                   "pool_size": len(pool), "trials_total": tl.total,
                   "trials_selection": tl.selection,
                   "kb_stats": st, "by_type": kb.by_type(),
                   "failure_reasons": kb.failure_reasons(),
                   "results": ranked, "elapsed_min": round(el / 60, 1)},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"Отчёт: {os.path.abspath(args.out)}")
    kb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
