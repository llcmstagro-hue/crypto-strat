"""
Ансамбли согласия — Путь Б.

Гипотеза прогона: одиночные механизмы (пробой, волатильность, межрынок) на
крипто-мейджорах затухли все и значимо. Но одиночный сигнал несёт и эдж, и
шум. Если шум у механизмов РАЗНОЙ природы некоррелирован, а эдж — общий,
то пересечение режет шум сильнее, чем эдж, и комбинация может держаться там,
где компоненты по отдельности уже нет.

Ключевой вопрос: **платит ли согласие там, где каждый поодиночке умер?**

Дисциплина строже обычной (условия трейдера):
  * не более 3 независимых сигналов в комбинации — жёсткий автоотсев Critic
    независимо от метрик;
  * у КАЖДОГО компонента объявлена рыночная логика ДО теста; «вместе дают PF»
    без объяснения почему — отсев;
  * лимит ширины поиска: не больше 36 гипотез за прогон, иначе планка барьера 5
    размывается для будущих кандидатов;
  * возврат к среднему не участвует — мёртв (ср. 1.29 барьера из 7).

Что здесь НЕ делается: не перебираются все сочетания всех блоков. Комбинации
перечислены поимённо, каждая отвечает на конкретный содержательный вопрос.
Декартово произведение дало бы сотни гипотез и подняло бы планку всем — при
том, что большинство сочетаний не имеет внятного объяснения.
"""

from __future__ import annotations

from ..engine.config import StrategyConfig
from ..validation.hypothesis import Hypothesis
from .score import Idea

MAX_HYPOTHESES = 36          # лимит ширины на прогон

ATR_STOP = {"type": "atr", "params": {"period": 14, "mult": 2.0}}
TRAIL = {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 240}}

# --------------------------------------------------------------------------- #
# ЛОГИКА КОМПОНЕНТОВ — объявлена ДО теста, требование Critic
# --------------------------------------------------------------------------- #
LOGIC = {
    "donchian_breakout": (
        "Выход цены за экстремум N баров запускает поток вынужденных заявок: "
        "выбиваются стопы противоположной стороны, входят ждавшие подтверждения. "
        "Механизм ценовой структуры, не зависит от режима волатильности."),
    "keltner_breakout": (
        "Движение вышло за границу, отмеренную в ATR от средней, то есть за "
        "рамки обычного шума ДАННОГО периода. Порог сам подстраивается под "
        "волатильность, в отличие от фиксированного окна Дончиана."),
    "squeeze_breakout": (
        "Волатильность кластеризуется и возвращается к среднему: аномально "
        "низкий разброс статистически сменяется высоким. Условие на РЕЖИМ "
        "рынка, а не на положение цены — источник информации иной природы."),
    "ttm_squeeze": (
        "Сжатие как вложение полос Боллинджера внутрь канала Кельтнера: два "
        "независимых измерителя разброса (стандартное отклонение против ATR). "
        "Сигнал на ОТПУСКАНИИ сжатия, направление по знаку моментума."),
    "nr_expansion": (
        "Самый узкий бар за N периодов — момент равновесия спроса и предложения. "
        "Выход за его границы означает, что равновесие нарушено. Механизм без "
        "индикаторов вообще, поэтому не делит источник шума с полосами и каналами."),
    "ref_momentum": (
        "Информация распространяется по крипторынку неравномерно: BTC впитывает "
        "новости первым, альты реагируют с задержкой из-за меньшей ликвидности "
        "и ограниченного внимания. Внешний по отношению к инструменту источник."),
    "ref_trend": (
        "Направление ведущего рынка как разрешение: у альтов нет собственного "
        "потока новостей сопоставимой силы, поэтому их движения против BTC чаще "
        "оказываются шумом. Условие внешнее, не выводится из цены инструмента."),
    "squeeze": (
        "Тот же режим волатильности, но в роли разрешения: сигнал засчитывается "
        "только если он возник ИЗ сжатия. Отвечает на вопрос «созрел ли рынок "
        "для движения», ортогональный вопросу «куда пошла цена»."),
    "htf_trend": (
        "Направление старшего таймфрейма: сигналы против него чаще оказываются "
        "проколами внутри коррекции. Информация с другого горизонта, а не ещё "
        "один взгляд на те же бары."),
}

SRC = {
    "breakout": ["Kaufman, Trading Systems and Methods", "система Дончиана"],
    "vol": ["TTM Squeeze (Carter)", "Bollinger on Bands — squeeze",
            "Crabel, NR7"],
    "xmkt": ["SSRN 3465924 Jia/Wu/Yan/Yin (межрыночный эффект)",
             "DergiPark: Bitcoin's lagged effect on altcoins"],
}


def _cfg(name, entry, filters, tf="4h") -> StrategyConfig:
    return StrategyConfig.from_dict({
        "name": name, "direction": "both", "timeframe": tf,
        "entry": entry, "stop": dict(ATR_STOP), "exit": dict(TRAIL),
        "filters": filters, "sizing": {"risk_pct": 0.01},
        "meta": {"path": "Б: ансамбли согласия"},
    })


def _consensus(components: list[str], window: int = 6, min_agree: int | None = None):
    return {"type": "consensus", "params": {
        "components": [{"type": c, "params": {}} for c in components],
        "window": window, "min_agree": min_agree or len(components)}}


def _idea(question: str, comps: list[str], sources: list[str],
          cross_market: bool = True) -> Idea:
    return Idea(
        rationale=(f"{question} Компоненты подобраны по РАЗНОЙ природе сигнала, "
                   f"чтобы их шум был некоррелирован: пересечение тогда режет "
                   f"шум сильнее, чем эдж."),
        sources=sources, cross_market=cross_market,
        component_logic={c: LOGIC[c] for c in comps})


def _h(name, entry, filters, grid, idea, primary=None, exclude=()):
    h = Hypothesis(_cfg(name, entry, filters), grid=grid,
                   source="Путь Б: согласие механизмов", notes=idea.rationale,
                   grid_cap=40)
    h.primary = primary
    h.exclude_symbols = exclude
    return h, idea


# --------------------------------------------------------------------------- #
# ПУЛ АНСАМБЛЕЙ
# --------------------------------------------------------------------------- #
def ensemble_pool() -> list[tuple[Hypothesis, Idea]]:
    P = []
    ALT = dict(primary="ETHUSDT", exclude=("BTCUSDT",))
    W = {"entry.params.window": [3, 6, 12]}
    STOP = {"stop.params.mult": [1.5, 2.0, 2.5]}

    # ---- 2 механизма: цена + волатильность ----
    P.append(_h("ens_donchian_x_squeeze",
                _consensus(["donchian_breakout", "squeeze_breakout"]), [],
                {**W, **STOP},
                _idea("Платит ли пробой уровня, случившийся ИЗ сжатия волатильности?",
                      ["donchian_breakout", "squeeze_breakout"],
                      SRC["breakout"] + SRC["vol"])))

    P.append(_h("ens_donchian_x_nr",
                _consensus(["donchian_breakout", "nr_expansion"]), [],
                {**W, **STOP},
                _idea("Совпадает ли пробой канала с выходом из самого узкого бара?",
                      ["donchian_breakout", "nr_expansion"],
                      SRC["breakout"] + SRC["vol"])))

    P.append(_h("ens_keltner_x_nr",
                _consensus(["keltner_breakout", "nr_expansion"]), [],
                {**W, **STOP},
                _idea("Волатильностный пробой плюс безиндикаторное сжатие: "
                      "подтверждают ли два разных измерителя друг друга?",
                      ["keltner_breakout", "nr_expansion"],
                      SRC["breakout"] + SRC["vol"])))

    # ---- 2 механизма: цена/волатильность + режим (фильтром) ----
    P.append(_h("ens_donchian_in_squeeze_regime",
                {"type": "donchian_breakout", "params": {"period": 20}},
                [{"type": "squeeze", "params": {"period": 20, "lookback": 120,
                                                "pct": 0.30, "squeezed": True}}],
                {"entry.params.period": [20, 30, 55],
                 "filters.0.params.pct": [0.20, 0.30, 0.40], **STOP},
                _idea("Тот же вопрос, но сжатие как РАЗРЕШЕНИЕ, а не как сигнал: "
                      "берём пробой только в сжатом рынке.",
                      ["donchian_breakout", "squeeze"],
                      SRC["breakout"] + SRC["vol"])))

    P.append(_h("ens_keltner_in_squeeze_regime",
                {"type": "keltner_breakout", "params": {"ema_period": 20,
                                                        "atr_period": 14, "mult": 2.0}},
                [{"type": "squeeze", "params": {"period": 20, "lookback": 120,
                                                "pct": 0.30, "squeezed": True}}],
                {"entry.params.mult": [1.5, 2.0, 2.5],
                 "filters.0.params.pct": [0.20, 0.30, 0.40], **STOP},
                _idea("Волатильностный пробой из сжатого режима.",
                      ["keltner_breakout", "squeeze"],
                      SRC["breakout"] + SRC["vol"])))

    # ---- 2 механизма: цена + внешний рынок ----
    P.append(_h("ens_donchian_x_btcmom",
                _consensus(["donchian_breakout", "ref_momentum"]), [],
                {**W, **STOP},
                _idea("Совпадает ли пробой альта с импульсом BTC? Внутренний "
                      "сигнал плюс внешний источник информации.",
                      ["donchian_breakout", "ref_momentum"],
                      SRC["breakout"] + SRC["xmkt"], cross_market=False), **ALT))

    P.append(_h("ens_squeeze_x_btcmom",
                _consensus(["squeeze_breakout", "ref_momentum"]), [],
                {**W, **STOP},
                _idea("Разжатие альта, совпавшее с импульсом BTC: режим "
                      "волатильности плюс внешний триггер.",
                      ["squeeze_breakout", "ref_momentum"],
                      SRC["vol"] + SRC["xmkt"], cross_market=False), **ALT))

    P.append(_h("ens_nr_x_btcmom",
                _consensus(["nr_expansion", "ref_momentum"]), [],
                {**W, **STOP},
                _idea("Выход из узкого бара, совпавший с импульсом BTC.",
                      ["nr_expansion", "ref_momentum"],
                      SRC["vol"] + SRC["xmkt"], cross_market=False), **ALT))

    # ---- 3 механизма: ПОТОЛОК сложности ----
    P.append(_h("ens_donchian_x_squeeze_btcfilter",
                _consensus(["donchian_breakout", "squeeze_breakout"]),
                [{"type": "ref_trend", "params": {"ref": "BTCUSDT", "ema_period": 50}}],
                {**W, "filters.0.params.ema_period": [30, 50, 100], **STOP},
                _idea("Полный ансамбль: цена + волатильность + разрешение "
                      "ведущего рынка. Три источника разной природы, потолок "
                      "допустимой сложности.",
                      ["donchian_breakout", "squeeze_breakout", "ref_trend"],
                      SRC["breakout"] + SRC["vol"] + SRC["xmkt"],
                      cross_market=False), **ALT))

    P.append(_h("ens_keltner_x_nr_btcfilter",
                _consensus(["keltner_breakout", "nr_expansion"]),
                [{"type": "ref_trend", "params": {"ref": "BTCUSDT", "ema_period": 50}}],
                {**W, "filters.0.params.ema_period": [30, 50, 100], **STOP},
                _idea("Тот же потолок, но другая пара внутренних механизмов.",
                      ["keltner_breakout", "nr_expansion", "ref_trend"],
                      SRC["breakout"] + SRC["vol"] + SRC["xmkt"],
                      cross_market=False), **ALT))

    P.append(_h("ens_donchian_squeeze_htf",
                _consensus(["donchian_breakout", "squeeze_breakout"]),
                [{"type": "htf_trend", "params": {"factor": 6, "ema_period": 50}}],
                {**W, "filters.0.params.ema_period": [30, 50, 100], **STOP},
                _idea("Три механизма без внешнего рынка: цена + волатильность + "
                      "старший таймфрейм. Контроль к межрыночной версии — "
                      "видно, что именно добавляет BTC.",
                      ["donchian_breakout", "squeeze_breakout", "htf_trend"],
                      SRC["breakout"] + SRC["vol"])))

    # ---- частичное согласие: 2 из 3 ----
    P.append(_h("ens_2of3_price_vol_xmkt",
                _consensus(["donchian_breakout", "squeeze_breakout", "ref_momentum"],
                           min_agree=2), [],
                {**W, **STOP},
                _idea("Смягчённое согласие: достаточно ДВУХ из трёх. Полное "
                      "совпадение трёх механизмов может быть слишком редким, "
                      "и выборка схлопнется — проверяем компромисс.",
                      ["donchian_breakout", "squeeze_breakout", "ref_momentum"],
                      SRC["breakout"] + SRC["vol"] + SRC["xmkt"],
                      cross_market=False), **ALT))

    P.append(_h("ens_2of3_breakouts_nr",
                _consensus(["donchian_breakout", "keltner_breakout", "nr_expansion"],
                           min_agree=2), [],
                {**W, **STOP},
                _idea("Два из трёх среди механизмов, близких по природе. "
                      "СОЗНАТЕЛЬНЫЙ контроль: если такое согласие работает не "
                      "хуже разнородного, значит дело не в независимости шума, "
                      "а просто в более редком входе.",
                      ["donchian_breakout", "keltner_breakout", "nr_expansion"],
                      SRC["breakout"] + SRC["vol"])))
    return P


def ensemble_seeds() -> tuple:
    """Семена для Evolution — по РАЗНООБРАЗИЮ состава, не по результату."""
    return ("ens_donchian_x_squeeze", "ens_donchian_x_btcmom",
            "ens_donchian_in_squeeze_regime")


# --------------------------------------------------------------------------- #
# ВНЕШНЯЯ СТРАТЕГИЯ: Courtney Smith, "The Conqueror" (гл. 4)
# --------------------------------------------------------------------------- #
CONQ_LOGIC = {
    "conqueror": (
        "Три фактора замеряют тренд на трёх разных горизонтах одним и тем же "
        "инструментом — ценой: положение относительно короткой средней (Cond1), "
        "наклон самой средней (Cond2) и долгосрочный моментум за 40 баров "
        "(Cond3). Вход только при согласии всех трёх: автор считает, что "
        "рассинхрон горизонтов означает неустойчивый тренд."),
}


def conqueror_pool(tf: str = "1d") -> list[tuple[Hypothesis, Idea]]:
    """Conqueror I и II — строго по первоисточнику, не по упрощённому .pine.

    Что именно взято из книги и почему это важно:

    ВХОД. Cond1 = close − SMA10; Cond2 = SMA10(t) − SMA10(t−10);
    Cond3 = close − close[−40]. Все три > 0 -> лонг, все три < 0 -> шорт.
    Сигнал в момент, когда выполнилось ПОСЛЕДНЕЕ условие.

    ВЫХОД — главная специфика метода, и именно её обычно упрощают:
      * трейлинг ведётся от ЭКСТРЕМАЛЬНОГО ЗАКРЫТИЯ с момента входа, а не от
        экстремального хая/лоу. Разница существенная: закрытия менее шумны,
        поэтому стоп не дёргается на проколах;
      * коэффициент ATR(40) сужается на треть при КАЖДОЙ смене знака ЛЮБОГО
        из трёх факторов, и это КУМУЛЯТИВНО: 2.00 -> 1.33 -> 0.89 -> 0.59...
        Это не бинарное переключение weak/strong. Смысл: чем больше факторов
        начало спорить друг с другом, тем меньше доверия тренду и тем ближе
        подтягивается стоп;
      * стоп двигается ТОЛЬКО в сторону прибыли, поэтому сужение может лишь
        подтянуть его, но не отпустить обратно.

    Conqueror I — базовый коэффициент 2.0 ATR. Conqueror II — 12.0 ATR,
    сверхдолгосрочная версия: стоп настолько широк, что позиция живёт до
    настоящего разворота, а вся работа ложится на кумулятивное сужение.

    АДАПТАЦИЯ ПОД КРИПТУ. Форекс-специфика первоисточника снята: пипсы и
    особые правила по JPY заменены на проценты и ATR, гэп-логика убрана
    (крипта торгуется 24/7, гэпов между барами нет). Риск — 0.5% equity на
    сделку, как аварийный предохранитель.

    ЧЕСТНАЯ ОГОВОРКА ОБ ИСПОЛНЕНИИ. Источник говорит «вход по закрытию бара».
    Движок исполняет по открытию следующего — сознательно консервативнее.
    В крипте это отличие копеечное (непрерывная торговля), а правило «не
    входить на баре, который только что увидел» — наш главный рубеж против
    lookahead, и ослаблять его ради буквы источника нельзя.
    """
    P = []
    src = ["Courtney Smith, How to Make a Living Trading Foreign Exchange, гл. 4",
           "первоисточник, НЕ упрощённая .pine-реализация"]

    for name, base in (("conqueror_I", 2.0), ("conqueror_II", 12.0)):
        entry = {"type": "conqueror", "params": {
            "sma_period": 10, "slope_lookback": 10, "mom_lookback": 40}}
        cfg = StrategyConfig.from_dict({
            "name": f"{name}_{tf}", "direction": "both", "timeframe": tf,
            "entry": entry,
            # аварийный стоп = начальный уровень трейлинга; сайзинг 0.5% equity
            "stop": {"type": "atr", "params": {"period": 40, "mult": base}},
            "exit": {"type": "conqueror_trail", "params": {
                "atr_period": 40, "base_mult": base, "narrow_factor": 2.0 / 3.0,
                "sma_period": 10, "slope_lookback": 10, "mom_lookback": 40}},
            "filters": [], "sizing": {"risk_pct": 0.005},
            "meta": {"source": "; ".join(src), "external": "Conqueror"},
        })
        # Сетка НАМЕРЕННО узкая: это ВНЕШНЯЯ стратегия с авторскими
        # параметрами, а не наша идея для подбора. Широкий перебор превратил бы
        # проверку чужого метода в поиск удачной его версии.
        grid = {
            "entry.params.mom_lookback": [20, 40, 80],
            "exit.params.mom_lookback": [20, 40, 80],
            "exit.params.base_mult": [base * 0.5, base, base * 1.5],
        }
        idea = Idea(
            rationale=(
                "Согласие трёх измерителей тренда на разных горизонтах плюс "
                "адаптивный трейлинг, сужающийся по мере того, как факторы "
                "начинают спорить между собой. Логика автора: рассинхрон "
                "горизонтов — ранний признак того, что тренд теряет силу, и "
                "реагировать надо сужением стопа, а не выходом."),
            sources=src, cross_market=True,
            component_logic=dict(CONQ_LOGIC))
        h = Hypothesis(cfg, grid=grid, source="внешняя стратегия: Courtney Smith",
                       notes=idea.rationale, grid_cap=40)
        P.append((h, idea))
    return P
