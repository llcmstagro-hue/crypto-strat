"""
Модуль [8] Evolution — БЕЗОПАСНАЯ ВЕРСИЯ (р.6.7).

Что здесь делается: порождение вариантов стратегии заменой ОДНОГО
структурного блока (выход, стоп, фильтр, направление) у осмысленного семени.

Что здесь НЕ делается и почему:

  ⛔ Не выбирается «лучший вариант по результату на истории». Лучший из N проб
     на одних данных — почти всегда везунчик, а не носитель эджа. Функции
     «взять победителя» в этом модуле просто нет.

  ⛔ Валидность НЕ наследуется. Вариант дончиана — это НЕ «улучшенный дончиан»,
     это ОТДЕЛЬНЫЙ кандидат, который проходит все семь барьеров с нуля. Тот
     факт, что родитель прошёл, не даёт варианту ни одного очка.

  ⛔ Каждый порождённый вариант увеличивает счётчик проб и тем самым поднимает
     планку значимости ВСЕМ кандидатам. Поэтому мутации перечислены поимённо,
     с рыночной логикой у каждой, а не порождаются декартовым произведением:
     ширина здесь оплачивается всеми.

По сути Evolution — это ещё один генератор кандидатов рядом с Collector,
а не оптимизатор.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..engine.config import StrategyConfig
from ..validation.hypothesis import Hypothesis
from .score import Idea


@dataclass
class Mutation:
    key: str
    rationale: str                 # ПОЧЕМУ эта замена осмысленна
    apply: object                  # callable(dict) -> dict (правит конфиг)
    grid: dict | None = None       # своя сетка вместо родительской части


def _set(d: dict, path: str, value):
    node = d
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value
    return d


# --------------------------------------------------------------------------- #
# КАТАЛОГ МУТАЦИЙ
# --------------------------------------------------------------------------- #
def mutation_catalog() -> list[Mutation]:
    def exit_trailing(c):
        c["exit"] = {"type": "trailing_atr",
                     "params": {"period": 14, "mult": 3.0, "max_bars": 240}}
        return c

    def exit_rr(c):
        c["exit"] = {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}}
        return c

    def exit_structure(c):
        c["exit"] = {"type": "structure", "params": {"ema_period": 50, "max_bars": 240}}
        return c

    def stop_structure(c):
        c["stop"] = {"type": "structure", "params": {"lookback": 20, "buffer": 0.0005}}
        return c

    def stop_channel(c):
        c["stop"] = {"type": "channel", "params": {"period": 20}}
        return c

    def add_htf(c):
        c["filters"] = c.get("filters", []) + [
            {"type": "htf_trend", "params": {"factor": 6, "ema_period": 50}}]
        return c

    def add_slope(c):
        c["filters"] = c.get("filters", []) + [
            {"type": "ema_slope", "params": {"period": 100, "lag": 20, "min_slope": 0.0}}]
        return c

    def add_atr_regime(c):
        c["filters"] = c.get("filters", []) + [
            {"type": "atr_regime", "params": {"min_pct": 0.005, "max_pct": 0.20}}]
        return c

    return [
        Mutation("exit_trailing",
                 "Пробой живёт правым хвостом: трейлинг не режет тренд, а даёт ему "
                 "развернуться. Проверяем, в хвосте ли эдж.",
                 exit_trailing, {"exit.params.mult": [2.5, 3.0, 4.0]}),
        Mutation("exit_rr",
                 "Обратная гипотеза: эдж не в хвосте, а в частом снятии умеренного "
                 "движения. Фиксированный R/R закрывает раньше и режет разброс.",
                 exit_rr, {"exit.params.rr": [1.5, 2.0, 3.0]}),
        Mutation("exit_structure",
                 "Выход не по цене и не по времени, а по слому структуры: сидим, "
                 "пока цена держится своей стороны средней.",
                 exit_structure, {"exit.params.ema_period": [30, 50, 100]}),
        Mutation("stop_structure",
                 "Стоп за последним свингом вместо ATR: риск привязан к структуре "
                 "рынка, а не к средней волатильности.",
                 stop_structure, {"stop.params.lookback": [10, 20, 40]}),
        Mutation("stop_channel",
                 "Стоп на противоположной границе канала: выход ровно там, где "
                 "тезис пробоя опровергнут.",
                 stop_channel, {"stop.params.period": [10, 20, 40]}),
        Mutation("filter_htf",
                 "Торговать только по направлению старшего ТФ. Гипотеза: пробои "
                 "против тренда старшего ТФ — это ложные проколы.",
                 add_htf, {"filters.-1.params.ema_period": [30, 50, 100]}),
        Mutation("filter_slope",
                 "Фильтр по наклону длинной EMA: отсекает боковик, где пробойные "
                 "сигналы вырождаются в пилу.",
                 add_slope, {"filters.-1.params.min_slope": [0.0, 0.01, 0.02]}),
        Mutation("filter_vol_regime",
                 "Отсечь мёртвую волатильность: при слишком узком ATR движение "
                 "после пробоя не окупает round-trip.",
                 add_atr_regime, {"filters.-1.params.min_pct": [0.004, 0.008, 0.015]}),
    ]


# --------------------------------------------------------------------------- #
def evolve(seed_cfg: StrategyConfig, seed_idea: Idea, seed_grid: dict,
           mutations: list[Mutation] | None = None,
           keep_seed_grid_keys: tuple = ("entry",)) -> list[tuple[Hypothesis, Idea]]:
    """Порождает независимых кандидатов из семени.

    keep_seed_grid_keys — какие части родительской сетки сохранить (обычно
    параметры входного блока: он и есть идея). Остальное задаёт мутация.
    """
    out = []
    for m in (mutations or mutation_catalog()):
        d = copy.deepcopy(seed_cfg.to_dict())
        d = m.apply(d)
        d["name"] = f"{seed_cfg.name}+{m.key}"
        d["meta"] = dict(d.get("meta", {}))
        d["meta"]["evolution"] = m.key
        d["meta"]["parent"] = seed_cfg.name
        try:
            cfg = StrategyConfig.from_dict(d)
        except Exception:
            continue

        grid = {k: v for k, v in seed_grid.items()
                if k.split(".")[0] in keep_seed_grid_keys}
        for path, values in (m.grid or {}).items():
            # '-1' в пути = только что добавленный фильтр
            if ".-1." in path:
                path = path.replace("filters.-1.", f"filters.{len(cfg.filters)-1}.")
            grid[path] = values

        idea = Idea(
            rationale=m.rationale,
            sources=list(seed_idea.sources) + ["Evolution (р.6.7), безопасная версия"],
            cross_market=seed_idea.cross_market,
        )
        out.append((Hypothesis(cfg, grid=grid,
                               source=f"Evolution от {seed_cfg.name}: {m.key}",
                               notes=m.rationale, grid_cap=40), idea))
    return out
