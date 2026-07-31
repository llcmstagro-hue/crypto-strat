"""
Перепрогон ЛУЧШИХ кандидатов на замеренном тарифе боевого счёта CFT/MT5.

ЧТО ЗА ТАРИФ И ОТКУДА ЧИСЛО
---------------------------
Замер по счёту: BTCUSDT.cft, 0.5 BTC @ 63814 (нотионал ~$31 900),
round-trip комиссия $20.73 -> 0.0650% round-trip = **3.25 bps на сторону**.
Исполнение на MT5 рыночное, гарантированный фил, разделения maker/taker нет —
ставка плоская для обеих сторон.

    модель          комиссия/сторона   слиппедж/сторона   round-trip
    легаси (1–10)      6.00 bps            9.0 bps          30.0 bps
    Bybit maker/taker  2.00 / 5.50         0 / 9.0          16.5 / 29.0
    **MT5 (замер)**    **3.25 bps**        **9.0 bps**      **24.5 bps**

Комиссия на MT5 действительно ниже биржевой тейкерной и сравнима с мейкерной.
Но round-trip падает только с 30.0 до 24.5 bps — на 18%, а не втрое, потому
что бóльшую часть издержки составляет проскальзывание, а не комиссия.
Ожидать от такого сдвига воскрешения мёртвого эджа не приходится; задача
прогона — проверить это, а не предположить.

ПРОСКАЛЬЗЫВАНИЕ НЕ СНИЖЕНО И НЕ БУДЕТ
-------------------------------------
Оставлено ровно 9 bps на сторону, как во всех прогонах. Более того, здесь
для этого есть дополнительное основание: торговля идёт через CFD-фид
проп-фирмы, а не напрямую в биржевом стакане. Спред там шире, глубина
неизвестна, и если реальность расходится с оценкой, то скорее в худшую
сторону. Ставка — замеренный факт; проскальзывание — оценка неизвестного,
и уточнять её в свою пользу нельзя. Пол вбит в конструктор конфига.

КОГО ГОНЯЕМ
-----------
Только тех, кто реально был близок: все, кто когда-либо брал 6/7, плюс
рекордсмен по робастности Conqueror II, плюс единственный, кто выходил в
положительный нетто на предыдущем этапе (ретест уровня), плюс order block
и Дончиан 55/20 стоп-приказами, названные трейдером поимённо.

Пулы НЕ переписываются заново — берутся те же самые сборщики гипотез, что и
в исходных прогонах, и у них подменяется только модель издержек. Так
исключается расхождение «переписал конфиг по памяти и сравнил не то с тем».
"""

from __future__ import annotations

from ..engine.config import MT5_COSTS, StrategyConfig
from ..validation.hypothesis import Hypothesis
from .ensembles import conqueror_pool, ensemble_pool, ensemble_seeds
from .evolution import evolve
from .pool import base_pool, new_classes_pool
from .score import Idea
from .smith import smith_pool

# Кандидаты по именам: все, кто брал 6/7, плюс названные трейдером.
# Имена с «+» — варианты Evolution, поэтому пул строится с Evolution.
WANTED = {
    "4h": [
        "order_block",                                # SMC, лимитный вход
        "level_retest",                               # единственный плюс на этапе 11
        "donchian_breakout+filter_htf",               # 6/7, заморожен на форвард
        "keltner_breakout+stop_channel",              # 6/7
        "xmkt_donchian_btc_filter+stop_structure",    # 6/7, лучший межрыночный
        "ens_keltner_x_nr_btcfilter",                 # 6/7, лучшая робастность H4
    ],
    "1d": [
        "smith_trend_analysis_1d",     # 6/7, лучший профиль за все прогоны
        "smith_channel_55_20_1d",      # Дончиан/«черепахи» 55/20 стоп-приказами
        "conqueror_II_1d",             # рекорд робастности 0.861
        "keltner_breakout+filter_vol_regime",   # 5/7, робастность 0.980
    ],
}

EVO_SEEDS = ("donchian_breakout", "keltner_breakout", "tsmom", "bb_meanrev",
             "order_block", "level_retest", "vol_squeeze_breakout",
             "vol_ttm_squeeze", "vol_nr_expansion", "xmkt_btc_spillover",
             "xmkt_btc_seesaw", "xmkt_donchian_btc_filter")


# --------------------------------------------------------------------------- #
# ЛОГИКА КОМПОНЕНТОВ для вариантов Evolution
# --------------------------------------------------------------------------- #
# Победители прошлых прогонов — это варианты вида «пробой + фильтр», у которых
# логика объявлена у РОДИТЕЛЯ, а не у самого варианта. Critic (справедливо)
# требует её у каждого компонента, иначе комбинация читается как «вместе дают
# PF». Собираем из уже написанных реестров; чего нет — дописываем здесь, но
# ДО теста и явно, а не задним числом.
_EXTRA_LOGIC = {
    "atr_regime": (
        "Коридор ATR/цена как условие о режиме: слишком тихий рынок не даёт "
        "движения, окупающего проезд, а экстремально шумный превращает любой "
        "уровень в лотерею — и там, и там трендовый вход теряет смысл. "
        "Условие о состоянии рынка, ортогональное вопросу о направлении."),
    "ma_side": (
        "Цена выше/ниже длинной средней — простейший механический признак "
        "того, на чьей стороне инициатива на горизонте, заметно большем "
        "торгового. Информация другого масштаба, а не второй взгляд на те же "
        "бары."),
    "ema_slope": (
        "Наклон длинной EMA: не «где цена относительно средней», а «куда "
        "средняя движется». Отличает состоявшийся тренд от бокового рынка, "
        "случайно оказавшегося по одну сторону от неё."),
}


def _merged_logic() -> dict:
    from .ensembles import LOGIC as ENS_LOGIC, CONQ_LOGIC
    from .recost import LOGIC as RC_LOGIC
    from .smith import LOGIC as SM_LOGIC
    from .fade import LOGIC as FD_LOGIC
    out = {}
    for reg in (ENS_LOGIC, CONQ_LOGIC, RC_LOGIC, SM_LOGIC, FD_LOGIC,
                _EXTRA_LOGIC):
        out.update(reg)
    return out


def _with_logic(cfg: StrategyConfig, idea: Idea) -> Idea:
    """Дополняет идею логикой каждого участвующего механизма.

    Если для какого-то компонента логики нет НИГДЕ — падаем громко. Тихо
    пропустить значило бы протащить в тест комбинацию без объяснения, ровно
    то, от чего Critic и поставлен.
    """
    from .score import component_types
    reg = _merged_logic()
    comps = component_types(cfg)
    missing = [c for c in comps if c not in reg]
    if missing:
        raise KeyError(f"{cfg.name}: нет объявленной логики у {missing} — "
                       f"допишите её ДО прогона, а не после")
    return Idea(rationale=idea.rationale, sources=list(idea.sources),
                cross_market=idea.cross_market,
                component_logic={c: reg[c] for c in comps})


def _to_mt5(h: Hypothesis) -> Hypothesis:
    """Тот же конфиг, та же сетка — другая модель издержек."""
    d = h.config.to_dict()
    d["costs"] = dict(MT5_COSTS)
    d["name"] = "mt5_" + d["name"]
    d.setdefault("meta", {})["cost_model"] = (
        "MT5/CFT: 3.25 bps/сторона (замер по счёту) + 9 bps проскальзывания")
    out = Hypothesis(StrategyConfig.from_dict(d), grid=dict(h.grid),
                     source="перепрогон на замеренном тарифе MT5/CFT",
                     notes=h.notes, grid_cap=h.grid_cap)
    out.primary = h.primary
    out.exclude_symbols = h.exclude_symbols
    return out


def _all_candidates(tf: str) -> dict[str, tuple[Hypothesis, Idea]]:
    """Все гипотезы, доступные на данном ТФ, включая варианты Evolution."""
    got: dict[str, tuple[Hypothesis, Idea]] = {}
    pools = [base_pool(), new_classes_pool(), ensemble_pool()]
    if tf == "1d":
        pools += [smith_pool("1d"), conqueror_pool("1d")]
    flat = [x for p in pools for x in p]
    # Evolution: победители прошлых прогонов — это варианты, а не базовые
    # гипотезы, и восстановить их можно только тем же генератором
    seeds = [(h, i) for h, i in flat
             if h.name in EVO_SEEDS + ensemble_seeds()]
    for h, idea in seeds:
        for vh, vi in evolve(h.config, idea, h.grid, primary=h.primary,
                             exclude_symbols=h.exclude_symbols):
            flat.append((vh, vi))
    for h, idea in flat:
        got.setdefault(h.name, (h, idea))
    return got


def mt5_pool(tf: str = "4h") -> list[tuple[Hypothesis, Idea]]:
    avail = _all_candidates(tf)
    out: list[tuple[Hypothesis, Idea]] = []
    missing = []
    for name in WANTED.get(tf, []):
        item = avail.get(name)
        if item is None:
            missing.append(name)
            continue
        h, idea = item
        mh = _to_mt5(h)
        out.append((mh, _with_logic(mh.config, idea)))
    if missing:
        # молча пропустить кандидата нельзя: отчёт тогда прочитается как
        # «проверили всех», хотя часть до теста не дошла
        print(f"[mt5_pool {tf}] НЕ НАЙДЕНЫ и НЕ ПРОВЕРЕНЫ: {', '.join(missing)}")
    return out
