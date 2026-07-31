"""
Перепрогон кандидатов на РЕАЛЬНОЙ модели исполнения (maker/taker Bybit).

ЧТО ИМЕННО ИЗМЕНИЛОСЬ И ПОЧЕМУ ЭТО ЗАКОННО
------------------------------------------
Прогоны 1–10 считали издержки плоско: 15 bps на сторону (6 комиссия + 9
проскальзывание), round-trip 30 bps, одинаково для любого исполнения. Для
стратегий, которые входят ПОКОЯЩИМСЯ ЛИМИТОМ, это втрое завышено: лимит
исполняется мейкером (0.02% на Bybit) и без проскальзывания — цена заявки
известна заранее.

Уточняется ТОЛЬКО КОМИССИЯ. Это факт о тарифе аккаунта, а не оценка.
Проскальзывание оставлено прежним (9 bps) — оно про неизвестное, и
подкручивать его «заодно» было бы тем же самообманом, что снижение порогов.

Новая модель:
    вход лимитом  = 2.0 bps                     (мейкер, без проскальзывания)
    вход рынком   = 5.5 + 9 = 14.5 bps          (тейкер)
    выход любой   = 5.5 + 9 = 14.5 bps          (тейкер: стоп/рынок)
    round-trip лимитной стратегии = 16.5 bps    (было 30)
    round-trip рыночной стратегии = 29.0 bps    (было 30 — почти без изменений)

Последнее число важнее, чем кажется: **для рыночных входов новая модель не
меняет НИЧЕГО**. Все вердикты по трендовому семейству, Conqueror и методам
Смита остаются как были — они входят по рынку или стоп-приказом. Оживить
уточнение тарифа может только тех, кто реально стоит в стакане.

ДВА УЖЕСТОЧЕНИЯ, ВВЕДЁННЫЕ ВМЕСТЕ С ПОСЛАБЛЕНИЕМ
------------------------------------------------
Скидка на комиссию не должна прийти одна, иначе перепрогон превратится в
поиск удобного результата:

1. **Поправка на очередь.** Раньше лимит считался исполненным, если цена
   КОСНУЛАСЬ уровня. Теперь рынок обязан пройти СКВОЗЬ заявку минимум на
   1 bp: мейкер стоит в стакане за чужим объёмом, и касание не гарантирует
   исполнения. Часть сделок из-за этого исчезает.
2. **Неисполненные лимиты не считаются сделками.** Это было и раньше, но у
   мейкерной версии fade это принципиально: заявка стоит на возврате к
   пробитому уровню, и если рынок не вернулся — сделки нет. Мейкерная версия
   даёт ЗАМЕТНО меньше сделок, чем рыночная, и это честный учёт, а не потеря.

ЧТО ПЕРЕПРОГОНЯЕТСЯ
-------------------
Только те, кто реально входит лимитом и упирался в издержки:
  * order block — H1 и H4 (лимит в зону);
  * FVG — H1 и H4 (лимит в разрыв);
  * ретест уровня — H4 (лимит на уровне);
  * fade ложного пробоя, МЕЙКЕРНАЯ версия — D1 и H4.

Чего здесь НЕТ и почему: пробои, Conqueror, методы Смита, ансамбли входят по
рынку или стоп-приказом. Их round-trip меняется с 30.0 на 29.0 bps — это
третий знак, и тратить на него пробы барьера 5 значило бы ухудшать планку
всем остальным ради заведомо нулевого эффекта.
"""

from __future__ import annotations

from ..engine.config import StrategyConfig
from ..validation.hypothesis import Hypothesis
from .score import Idea

# Реальные ставки аккаунта. Прописаны явно в каждом конфиге, чтобы в базе
# знаний было видно, на какой модели считалась запись.
REAL_COSTS = {
    "model": "maker_taker",
    "maker_fee_bps": 2.0,
    "taker_fee_bps": 5.5,
    "slip_bps": 9.0,
    "maker_through_bps": 1.0,
    "funding_default_8h": 0.0001,
}

SRC = ["перерасчёт кандидатов на реальных ставках Bybit (maker 0.02% / "
       "taker 0.055%), проскальзывание не менялось",
       "методология трейдера (р.6.5 п.4)"]

LOGIC = {
    "order_block": (
        "После слома структуры цена возвращается к последней противоположной "
        "свече импульса — зоне, где остались неисполненные заявки крупного "
        "участника. Вход покоящимся лимитом В зону: мы не гонимся за ценой, а "
        "ждём её у себя, поэтому исполнение мейкерное по построению."),
    "fvg": (
        "Трёхсвечный разрыв — след движения, прошедшего без встречной "
        "ликвидности. Цена возвращается заполнить неэффективность. Вход "
        "лимитом в разрыв, то есть тоже покоящейся заявкой."),
    "level_retest": (
        "Пробитый уровень меняет роль: бывшее сопротивление становится "
        "поддержкой. Вход на ретесте лимитом даёт лучшую цену и более близкий "
        "стоп, чем вход на самом пробое."),
    "false_breakout_fade": (
        "Цена сходила за уровень предыдущего дня, где стоят стопы, собрала их "
        "и не смогла там закрыться. Мейкерная версия ждёт ВОЗВРАТА к самому "
        "пробитому уровню покоящейся заявкой: сделка есть только если рынок "
        "вернулся, а не «мы предположительно вошли по закрытию»."),
    "htf_trend": (
        "Направление старшего таймфрейма: прокол против основного тренда чаще "
        "оказывается откатом внутри тренда, который продолжится, а прокол по "
        "тренду — исчерпанием контртрендовой попытки."),
}


def _idea(rationale: str, comps: list[str]) -> Idea:
    return Idea(rationale=rationale, sources=list(SRC), cross_market=True,
                component_logic={c: LOGIC[c] for c in comps})


def _cfg(name, entry, stop, exit_, filters, tf) -> StrategyConfig:
    return StrategyConfig.from_dict({
        "name": f"{name}_{tf}", "direction": "both", "timeframe": tf,
        "entry": entry, "stop": stop, "exit": exit_, "filters": filters,
        "sizing": {"risk_pct": 0.01}, "costs": dict(REAL_COSTS),
        "meta": {"source": "; ".join(SRC),
                 "cost_model": "maker 2 / taker 5.5 / slip 9 bps"},
    })


def _h(cfg, grid, idea) -> tuple[Hypothesis, Idea]:
    return (Hypothesis(cfg, grid=grid,
                       source="перерасчёт на реальных издержках",
                       notes=idea.rationale, grid_cap=40), idea)


BLOCK_STOP = {"type": "block", "params": {}}
TRAIL = {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0,
                                            "max_bars": 240}}


def recost_pool(tf: str = "4h") -> list[tuple[Hypothesis, Idea]]:
    P: list[tuple[Hypothesis, Idea]] = []

    # ---- лимитные входы методологии трейдера: сетки те же, что в исходных
    # прогонах, чтобы сравнение было «при прочих равных» ----
    if tf in ("1h", "4h"):
        P.append(_h(_cfg(
            "rc_order_block",
            {"type": "order_block", "params": {"swing_left": 3, "swing_right": 3,
                                               "ob_max_age": 60,
                                               "stop_buffer": 0.0005}},
            dict(BLOCK_STOP),
            {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}}, [], tf),
            {"entry.params.swing_left": [2, 3, 4],
             "entry.params.ob_max_age": [40, 60, 80],
             "exit.params.rr": [1.5, 2.0, 3.0]},
            _idea("Order block с мейкерным входом на реальных ставках. Именно "
                  "он раньше упирался в издержки: валовой эдж 0.086–0.137% "
                  "нотионала против round-trip 0.30%. При 16.5 bps порог "
                  "проезда падает вдвое.", ["order_block"])))

        P.append(_h(_cfg(
            "rc_fvg",
            {"type": "fvg", "params": {"min_size_pct": 0.002, "max_age": 40,
                                       "stop_buffer": 0.0005}},
            dict(BLOCK_STOP),
            {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}}, [], tf),
            {"entry.params.min_size_pct": [0.001, 0.002, 0.004],
             "entry.params.max_age": [20, 40, 80],
             "exit.params.rr": [1.5, 2.0, 3.0]},
            _idea("FVG с мейкерным входом в разрыв на реальных ставках.",
                  ["fvg"])))

    if tf == "4h":
        P.append(_h(_cfg(
            "rc_level_retest",
            {"type": "level_retest", "params": {"swing_left": 3, "swing_right": 3,
                                                "max_age": 30,
                                                "stop_atr_mult": 1.5,
                                                "atr_period": 14}},
            dict(BLOCK_STOP), dict(TRAIL), [], tf),
            {"entry.params.max_age": [20, 30, 60],
             "entry.params.stop_atr_mult": [1.0, 1.5, 2.0],
             "exit.params.mult": [2.5, 3.0, 3.5]},
            _idea("Ретест пробитого уровня с мейкерным входом на реальных "
                  "ставках.", ["level_retest"])))

    # ---- мейкерная версия fade: заявка ждёт ВОЗВРАТА к пробитому уровню ----
    if tf in ("1d", "4h"):
        max_bars = 10 if tf == "1d" else 30
        factor = 7 if tf == "1d" else 6
        FADE_ENTRY = {"type": "false_breakout_fade",
                      "params": {"stop_buffer": 0.0005, "entry_mode": "limit",
                                 "max_age": 3}}
        RR = {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": max_bars}}
        OPP = {"type": "level_target", "params": {"max_bars": max_bars}}
        HTF = [{"type": "htf_trend", "params": {"factor": factor,
                                                "ema_period": 50}}]

        note = ("Fade ложного пробоя, МЕЙКЕРНАЯ версия: покоящаяся заявка на "
                "возврате к пробитому уровню предыдущего дня. Сделка "
                "засчитывается, только если рынок реально вернулся и прошёл "
                "сквозь заявку; не вернулся — сделки нет. ")

        P.append(_h(_cfg("rc_fade_rr", dict(FADE_ENTRY), dict(BLOCK_STOP),
                         dict(RR), [], tf),
                    {"exit.params.rr": [1.5, 2.0, 2.5],
                     "entry.params.max_age": [2, 3, 5]},
                    _idea(note + "Выход по фиксированному R/R.",
                          ["false_breakout_fade"])))

        P.append(_h(_cfg("rc_fade_opposite", dict(FADE_ENTRY), dict(BLOCK_STOP),
                         dict(OPP), [], tf),
                    {"entry.params.max_age": [2, 3, 5],
                     "exit.params.max_bars": [max_bars // 2, max_bars,
                                              max_bars * 2]},
                    _idea(note + "Выход у противоположной границы вчерашнего "
                                 "дня.", ["false_breakout_fade"])))

        P.append(_h(_cfg("rc_fade_rr_htf", dict(FADE_ENTRY), dict(BLOCK_STOP),
                         dict(RR), list(HTF), tf),
                    {"exit.params.rr": [1.5, 2.0, 2.5],
                     "filters.0.params.ema_period": [30, 50, 100]},
                    _idea(note + "Плюс фильтр старшего тренда — версия, у "
                                 "которой валовой эдж был наибольшим "
                                 "(+0.105% нотионала).",
                          ["false_breakout_fade", "htf_trend"])))

        P.append(_h(_cfg("rc_fade_opposite_htf", dict(FADE_ENTRY),
                         dict(BLOCK_STOP), dict(OPP), list(HTF), tf),
                    {"entry.params.max_age": [2, 3, 5],
                     "filters.0.params.ema_period": [30, 50, 100]},
                    _idea(note + "Возврат к противоположной границе плюс "
                                 "фильтр старшего тренда.",
                          ["false_breakout_fade", "htf_trend"])))
    return P
