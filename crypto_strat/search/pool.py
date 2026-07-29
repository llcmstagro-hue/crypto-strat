"""
Пул гипотез для широкого поиска на H4 + метаданные под Research Score (р.6.7).

К каждой гипотезе прилагается Idea: рыночная логика (ПОЧЕМУ это должно
работать), источники и признак кросс-рыночности. Без этого Research Score
считать нечем, а без него любая идея выглядит одинаково достойной дорогого
бэктеста.

Про состав. Базовые 8 — те же, что гонялись раньше (академия, классика,
открытый код). Добавлены четыре из методологии трейдера (order block, FVG,
ретест уровня, дивергенция). По р.6.5 свои сетапы — калибровочный эталон, а не
сырьё Фазы 1, но трейдер явно попросил прогнать order block, поэтому вся семья
включена и проходит фильтр НА ОБЩИХ ОСНОВАНИЯХ, без поблажек.

Все параметры — в барах H4. Один бар H4 = 4 часа, поэтому период 20 это ~3.3
суток, а lookback 180 — месяц.
"""

from __future__ import annotations

from ..engine.config import StrategyConfig
from ..validation.hypothesis import Hypothesis
from .score import Idea


def _mk(name, entry, stop, exit_, filters, grid, idea: Idea,
        direction="both", cap=40) -> tuple[Hypothesis, Idea]:
    cfg = StrategyConfig.from_dict({
        "name": name, "direction": direction, "timeframe": "4h",
        "entry": entry, "stop": stop, "exit": exit_, "filters": filters,
        "sizing": {"risk_pct": 0.01},
        "meta": {"source": "; ".join(idea.sources), "rationale": idea.rationale},
    })
    src = idea.sources[0] if idea.sources else ""
    return Hypothesis(cfg, grid=grid, source=src, notes=idea.rationale,
                      grid_cap=cap), idea


ATR_STOP = {"type": "atr", "params": {"period": 14, "mult": 2.0}}
TRAIL = {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 240}}


def base_pool() -> list[tuple[Hypothesis, Idea]]:
    P = []

    # ---------------- пробой / тренд ----------------
    P.append(_mk(
        "donchian_breakout",
        {"type": "donchian_breakout", "params": {"period": 20}}, ATR_STOP, TRAIL, [],
        {"entry.params.period": [20, 30, 55], "stop.params.mult": [1.5, 2.0, 2.5],
         "exit.params.mult": [2.5, 3.0, 3.5]},
        Idea(rationale=(
            "Цена, вышедшая за экстремум N последних баров, чаще продолжает движение, "
            "чем разворачивается: пробой выбивает стопы противоположной стороны и "
            "запускает поток вынужденных заявок. Плюс участники, ждавшие подтверждения, "
            "входят именно на пробое. Механизм не зависит от инструмента."),
             sources=["Kaufman, Trading Systems and Methods", "система Дончиана/«черепах»"],
             cross_market=True)))

    P.append(_mk(
        "keltner_breakout",
        {"type": "keltner_breakout", "params": {"ema_period": 20, "atr_period": 14, "mult": 2.0}},
        ATR_STOP, TRAIL, [],
        {"entry.params.mult": [1.5, 2.0, 2.5], "entry.params.ema_period": [20, 40],
         "stop.params.mult": [1.5, 2.0, 2.5]},
        Idea(rationale=(
            "Волатильностный пробой: граница отмеряется в ATR от средней, а не по "
            "ценовому экстремуму. Сигнал возникает, когда движение выходит за рамки "
            "обычного шума ДАННОГО периода, что автоматически подстраивается под "
            "смену волатильности — в отличие от фиксированного окна Дончиана."),
             sources=["Kaufman, Trading Systems and Methods", "каналы Кельтнера"],
             cross_market=True)))

    P.append(_mk(
        "tsmom",
        {"type": "tsmom", "params": {"lookback": 180}},
        {"type": "atr", "params": {"period": 14, "mult": 3.0}},
        {"type": "time", "params": {"bars": 180, "max_bars": 180}}, [],
        {"entry.params.lookback": [90, 180, 360], "stop.params.mult": [2.5, 3.0, 4.0],
         "exit.params.bars": [90, 180, 360]},
        Idea(rationale=(
            "Time-series momentum: знак собственной доходности за прошлый период "
            "предсказывает знак будущей. Объясняется медленным впитыванием информации "
            "и поведением следующих за трендом участников. Документирован на десятках "
            "рынков и классов активов за 100+ лет."),
             sources=["SSRN 2089463 Moskowitz/Ooi/Pedersen, Time Series Momentum",
                      "SSRN 4675565 Han/Kang/Ryu, крипто с издержками"],
             cross_market=True)))

    P.append(_mk(
        "ma_cross",
        {"type": "ma_cross", "params": {"fast": 20, "slow": 50}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "structure", "params": {"ema_period": 50, "max_bars": 480}}, [],
        {"entry.params.fast": [10, 20, 30], "entry.params.slow": [50, 100],
         "stop.params.mult": [2.0, 2.5, 3.0]},
        Idea(rationale=(
            "Пересечение быстрой и медленной средней — простейший детектор смены "
            "режима цены. Включён сознательно как СЛАБАЯ гипотеза: если такое "
            "проходит фильтр, это повод перепроверить сам фильтр."),
             sources=["Carver, Systematic Trading", "Kaufman"],
             cross_market=True)))

    # ---------------- возврат к среднему ----------------
    P.append(_mk(
        "bb_meanrev",
        {"type": "bollinger_meanrev", "params": {"period": 20, "k": 2.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "fixed_rr", "params": {"rr": 1.5, "max_bars": 96}}, [],
        {"entry.params.k": [2.0, 2.5, 3.0], "entry.params.period": [20, 40],
         "exit.params.rr": [1.0, 1.5, 2.0]},
        Idea(rationale=(
            "Выход цены далеко за полосу — часто следствие разовой ликвидации или "
            "паники, а не новой информации. Такое движение статистически "
            "возвращается к средней. Работает в боковике, ломается в тренде."),
             sources=["freqtrade Strategy002 (ядро)", "Bollinger, Bollinger on Bands"],
             cross_market=True)))

    P.append(_mk(
        "bb_meanrev_rsi",
        {"type": "bollinger_meanrev", "params": {"period": 20, "k": 2.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "fixed_rr", "params": {"rr": 1.5, "max_bars": 96}},
        [{"type": "rsi_bound", "params": {"period": 14, "min": 0, "max": 35}}],
        {"entry.params.k": [2.0, 2.5, 3.0], "exit.params.rr": [1.0, 1.5, 2.0],
         "filters.0.params.max": [30, 35, 40]},
        Idea(rationale=(
            "Тот же возврат к среднему, но с требованием подтверждённой "
            "перепроданности: фильтр отсекает пробои полосы, происходящие внутри "
            "сильного тренда, где возврата не будет."),
             sources=["freqtrade Strategy002", "Bollinger"],
             cross_market=True)))

    P.append(_mk(
        "rsi2_connors",
        {"type": "rsi_threshold", "params": {"period": 2, "oversold": 5.0, "overbought": 95.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "structure", "params": {"ema_period": 5, "max_bars": 72}},
        [{"type": "ma_side", "params": {"period": 200, "kind": "sma"}}],
        {"entry.params.oversold": [5.0, 10.0], "entry.params.period": [2, 3],
         "exit.params.ema_period": [5, 10], "filters.0.params.period": [100, 200]},
        Idea(rationale=(
            "Краткосрочная перепроданность ВНУТРИ восходящего тренда: RSI(2) ловит "
            "локальную панику, фильтр SMA-200 гарантирует, что покупаем откат в "
            "тренде, а не ловим падающий нож. Классика возврата к среднему."),
             sources=["StockCharts ChartSchool, RSI(2)", "QuantifiedStrategies, Connors"],
             cross_market=True)))

    # ---------------- методология трейдера (SMC) ----------------
    P.append(_mk(
        "order_block",
        {"type": "order_block", "params": {"swing_left": 3, "swing_right": 3,
                                           "ob_max_age": 60, "stop_buffer": 0.0005}},
        {"type": "block", "params": {}},
        {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}}, [],
        {"entry.params.swing_left": [2, 3, 4], "entry.params.ob_max_age": [40, 60, 80],
         "exit.params.rr": [1.5, 2.0, 3.0]},
        Idea(rationale=(
            "После слома структуры цена часто возвращается к последней "
            "противоположной свече импульса — зоне, где остались неисполненные "
            "заявки крупного участника. Вход лимитом в зону даёт близкий стоп "
            "за её границей."),
             sources=["методология трейдера (р.6.5 п.4)", "SMC / order block"],
             cross_market=True)))

    P.append(_mk(
        "fvg",
        {"type": "fvg", "params": {"min_size_pct": 0.002, "max_age": 40, "stop_buffer": 0.0005}},
        {"type": "block", "params": {}},
        {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}}, [],
        {"entry.params.min_size_pct": [0.001, 0.002, 0.004],
         "entry.params.max_age": [20, 40, 80], "exit.params.rr": [1.5, 2.0, 3.0]},
        Idea(rationale=(
            "Трёхсвечный разрыв — след движения, прошедшего без встречной "
            "ликвидности. Гипотеза: цена возвращается заполнить неэффективность, "
            "после чего продолжает исходное направление."),
             sources=["методология трейдера", "SMC / fair value gap"],
             cross_market=True)))

    P.append(_mk(
        "level_retest",
        {"type": "level_retest", "params": {"swing_left": 3, "swing_right": 3,
                                            "max_age": 30, "stop_atr_mult": 1.5,
                                            "atr_period": 14}},
        {"type": "block", "params": {}}, TRAIL, [],
        {"entry.params.max_age": [20, 30, 60], "entry.params.stop_atr_mult": [1.0, 1.5, 2.0],
         "exit.params.mult": [2.5, 3.0, 3.5]},
        Idea(rationale=(
            "Пробитый уровень меняет роль: бывшее сопротивление становится "
            "поддержкой. Вход на ретесте даёт лучшую цену, чем вход на самом "
            "пробое, и более близкий стоп. Классика горизонтальных уровней."),
             sources=["классика технического анализа", "методология трейдера"],
             cross_market=True)))

    P.append(_mk(
        "rsi_divergence",
        {"type": "rsi_divergence", "params": {"swing_left": 3, "swing_right": 3,
                                              "rsi_period": 14, "max_gap": 40}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}}, [],
        {"entry.params.rsi_period": [7, 14, 21], "entry.params.max_gap": [20, 40, 60],
         "exit.params.rr": [1.5, 2.0, 3.0]},
        Idea(rationale=(
            "Новый ценовой экстремум без подтверждения импульсом означает "
            "истощение движения: продавцы/покупатели теряют силу. Сигнал "
            "разворота на подтверждённых свингах."),
             sources=["методология трейдера", "классика осцилляторов"],
             cross_market=True)))

    return P


def evolution_seeds(pool: list[tuple[Hypothesis, Idea]],
                    names: tuple = ("donchian_breakout", "keltner_breakout", "tsmom",
                                    "bb_meanrev", "order_block", "level_retest")):
    """Семена для Evolution. Отбор семян по РАЗНООБРАЗИЮ механизмов
    (пробой / волатильность / моментум / возврат / SMC), а НЕ по результату
    на истории: выбирать семена по прошлому результату — это и есть та самая
    ядовитая версия Evolution из р.6.7."""
    return [(h, i) for h, i in pool if h.name in names]
