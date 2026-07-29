"""
Модуль [1b] — очередь гипотез, извлечённых из внешних источников (р.6.5).

Правила отбора сырья, которым следовал поисковик:

  * Оценка ТОЛЬКО по двум признакам: (а) есть ли чёткие механические правила,
    переводимые в конфиг р.6; (б) не дубликат ли уже стоящего в очереди.
  * Обещанная доходность ИГНОРИРУЕТСЯ полностью. «Sharpe 2.17», «90% winrate»,
    «5% ROI за 60 свечей» — это не информация, а приманка. Ни одна цифра
    доходности из источника не переносится сюда и не влияет на приоритет.
  * Источник не даёт поблажек: идея из SSRN и идея из freqtrade проходят
    один и тот же фильтр р.4.

ВАЖНОЕ ПРЕОБРАЗОВАНИЕ ПРИ ПЕРЕНОСЕ. Стратегии из freqtrade написаны на 5m
и навешивают 6-8 условий сразу. Переносить их «как есть» нельзя по двум
причинам: (1) проект сознательно не опускается ниже H1; (2) 7 условий на
выборке в пару сотен сделок — это прямой красный флаг р.4, и наша же
калибровка (случай C2) показывает, что такая конструкция умирает на барьере
минимума сделок. Поэтому из каждой берётся ЯДРО механики — 2-3 условия,
несущие идею, — и переносится на H1/H4. Это осознанное упрощение, а не
искажение: если эдж есть только в конъюнкции семи условий на 5m, он и не
переживёт честной валидации.

Сетки узкие намеренно: каждая лишняя комбинация поднимает планку барьера 5
для ВСЕХ гипотез разом.
"""

from __future__ import annotations

from ..engine.config import StrategyConfig
from ..validation.hypothesis import Hypothesis

DEFAULT_SIZING = {"risk_pct": 0.01}


def _h(name, entry, stop, exit_, filters, grid, source, notes,
       direction="both", tf="1h") -> Hypothesis:
    cfg = StrategyConfig.from_dict({
        "name": name, "direction": direction, "timeframe": tf,
        "entry": entry, "stop": stop, "exit": exit_, "filters": filters,
        "sizing": dict(DEFAULT_SIZING),
        "meta": {"source": source, "notes": notes},
    })
    return Hypothesis(cfg, grid=grid, source=source, notes=notes, grid_cap=40)


# --------------------------------------------------------------------------- #
# ФАЗА 1 — АКАДЕМИЯ (р.6.5, источник №1)
# --------------------------------------------------------------------------- #
def tsmom() -> Hypothesis:
    """Time-series momentum.

    Правило первоисточника (Moskowitz, Ooi, Pedersen 2012): знак собственной
    доходности за lookback предсказывает направление; лонг при положительной,
    шорт при отрицательной; удержание фиксированное. Крипто-адаптации
    (Han/Kang/Ryu, SSRN 4675565) подтверждают, что TS-моментум в крипте
    держится лучше, чем cross-sectional, но подчёркивают: при честных
    транзакционных издержках значимость большинства портфелей исчезает —
    ровно то, что здесь и проверяется.

    Перенос: месячные горизонты статьи -> часовые бары. lookback 168/336/720 H1
    = примерно неделя / две / месяц.
    """
    return _h(
        "acad_tsmom",
        {"type": "tsmom", "params": {"lookback": 336}},
        {"type": "atr", "params": {"period": 14, "mult": 3.0}},
        {"type": "time", "params": {"bars": 336, "max_bars": 336}},
        [],
        {"entry.params.lookback": [168, 336, 720],
         "stop.params.mult": [2.5, 3.0, 4.0],
         "exit.params.bars": [168, 336, 720]},
        "SSRN 2089463 (Moskowitz/Ooi/Pedersen, Time Series Momentum); "
        "SSRN 4675565 (Han/Kang/Ryu, крипто, с издержками)",
        "Механика полностью формализуема. Волатильностное масштабирование позиции "
        "из статьи НЕ переносим: у нас риск фиксирован в % на сделку под CFT.",
    )


def tsmom_htf_filter() -> Hypothesis:
    """TSMOM + фильтр тренда старшего ТФ.

    Не дубликат предыдущей: проверяется отдельная гипотеза р.6.5 п.4 —
    эдж часто рождается из КОМБИНАЦИИ известных блоков. Здесь тот же сигнал,
    но торгуется только по направлению старшего ТФ.
    """
    return _h(
        "acad_tsmom_htf",
        {"type": "tsmom", "params": {"lookback": 336}},
        {"type": "atr", "params": {"period": 14, "mult": 3.0}},
        {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 480}},
        [{"type": "htf_trend", "params": {"factor": 4, "ema_period": 50}}],
        {"entry.params.lookback": [168, 336, 720],
         "exit.params.mult": [2.5, 3.0, 4.0],
         "filters.0.params.ema_period": [30, 50]},
        "SSRN 2089463 + комбинаторика блоков (р.6.5 п.4)",
        "4 правила — в пределах разумного (красный флаг начинается с >6).",
    )


# --------------------------------------------------------------------------- #
# ФАЗА 1 — КЛАССИКА СИСТЕМНОЙ ТОРГОВЛИ (р.6.5, источник №3)
# --------------------------------------------------------------------------- #
def donchian_turtle() -> Hypothesis:
    """Пробой канала Дончиана (каркас «черепах»).

    Правила общеизвестны и однозначны: вход при пробое канала за N баров,
    стоп в ATR, выход по трейлингу. Пирамидинг и правило «пропусти сделку
    после выигрышной» НЕ переносим: первое противоречит фиксированному риску
    под CFT, второе вводит зависимость от истории сделок, которую честно
    валидировать сложнее.
    """
    return _h(
        "classic_donchian_breakout",
        {"type": "donchian_breakout", "params": {"period": 20}},
        {"type": "atr", "params": {"period": 14, "mult": 2.0}},
        {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 240}},
        [],
        {"entry.params.period": [20, 30, 55],
         "stop.params.mult": [1.5, 2.0, 2.5],
         "exit.params.mult": [2.5, 3.0, 3.5]},
        "Kaufman, Trading Systems and Methods; система Дончиана/«черепах»",
        "Тот же блок, что и в позитивном контроле калибровки. Здесь он "
        "проходит фильтр НА ОБЩИХ ОСНОВАНИЯХ как обычная гипотеза.",
    )


def keltner_vol_breakout() -> Hypothesis:
    """Волатильностный пробой канала Кельтнера.

    Отличается от Дончиана по природе: там пробой ЦЕНОВОГО экстремума,
    здесь — выход за границу, отмеренную в ATR от EMA. Не дубликат.
    """
    return _h(
        "classic_keltner_breakout",
        {"type": "keltner_breakout", "params": {"ema_period": 20, "atr_period": 14,
                                                "mult": 2.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.0}},
        {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 240}},
        [],
        {"entry.params.mult": [1.5, 2.0, 2.5],
         "entry.params.ema_period": [20, 40],
         "stop.params.mult": [1.5, 2.0, 2.5]},
        "Kaufman; каналы Кельтнера как волатильностный пробой",
        "",
    )


def ma_cross_trend() -> Hypothesis:
    """Пересечение двух EMA — базовое трендследование.

    Включено сознательно как «слабая» гипотеза: если фильтр пропускает такое
    на реальных данных, это повод проверить сам фильтр, а не радоваться.
    """
    return _h(
        "classic_ma_cross",
        {"type": "ma_cross", "params": {"fast": 20, "slow": 50}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "structure", "params": {"ema_period": 50, "max_bars": 480}},
        [],
        {"entry.params.fast": [10, 20, 30],
         "entry.params.slow": [50, 100],
         "stop.params.mult": [2.0, 2.5, 3.0]},
        "Carver, Systematic Trading; Kaufman",
        "",
    )


# --------------------------------------------------------------------------- #
# ФАЗА 1 — ОТКРЫТЫЙ КОД (р.6.5, источник №2)
# --------------------------------------------------------------------------- #
def bb_rsi_meanrev() -> Hypothesis:
    """Возврат к среднему: выход за нижнюю полосу Боллинджера + перепроданность.

    Источник: freqtrade-strategies, Strategy002 (5m). В оригинале конъюнкция
    из четырёх условий входа (RSI<30, stochastic slowk<20, close ниже нижней
    полосы BB(20,2), свечной паттерн «молот») и выход по SAR + Fisher RSI.

    Перенесено ЯДРО: пробой нижней полосы BB как вход + RSI-коридор как фильтр,
    на H1. Свечной паттерн и стохастик отброшены осознанно — они добавляют
    условий, но не идею, а на нашей выборке каждое лишнее условие режет
    статистику (р.4, красные флаги). minimal_roi и stoploss -10% оригинала
    не переносятся: у нас выход задаётся блоком, а риск — сайзингом под CFT.
    """
    return _h(
        "oss_bb_rsi_meanrev",
        {"type": "bollinger_meanrev", "params": {"period": 20, "k": 2.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "fixed_rr", "params": {"rr": 1.5, "max_bars": 96}},
        [{"type": "rsi_bound", "params": {"period": 14, "min": 0, "max": 35}}],
        {"entry.params.k": [2.0, 2.5, 3.0],
         "exit.params.rr": [1.0, 1.5, 2.0],
         "filters.0.params.max": [30, 35, 40]},
        "github.com/freqtrade/freqtrade-strategies — Strategy002 (ядро механики)",
        "Оригинал 5m/7 условий сведён к H1/3 правилам. Обещания доходности "
        "оригинала (minimal_roi 5%) проигнорированы.",
    )


def rsi2_connors() -> Hypothesis:
    """Connors RSI-2 с трендовым фильтром по SMA-200.

    Правила источника: лонг при RSI(2) < 5 и цене выше SMA-200; шорт при
    RSI(2) > 95 и цене ниже SMA-200; выход при возврате RSI выше 65 либо по
    короткой скользящей. Правила механические и однозначные — годное сырьё
    независимо от того, что об эффективности пишут.

    Перенос: дневная схема оригинала -> H1 и H4 (проверяем оба ТФ отдельно).
    """
    return _h(
        "oss_rsi2_connors",
        {"type": "rsi_threshold", "params": {"period": 2, "oversold": 5.0,
                                             "overbought": 95.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "structure", "params": {"ema_period": 5, "max_bars": 72}},
        [{"type": "ma_side", "params": {"period": 200, "kind": "sma"}}],
        {"entry.params.oversold": [5.0, 10.0],
         "entry.params.period": [2, 3],
         "exit.params.ema_period": [5, 10],
         "filters.0.params.period": [100, 200]},
        "QuantifiedStrategies / StockCharts ChartSchool — RSI(2), Larry Connors",
        "Симметричный шорт добавлен нами (в оригинале акцент на лонгах); "
        "фильтр SMA-200 сохранён как в источнике.",
    )


def bb_meanrev_no_filter() -> Hypothesis:
    """Тот же возврат к среднему, но БЕЗ фильтра RSI.

    Нужна как контроль к oss_bb_rsi_meanrev: если версия с фильтром проходит,
    а без фильтра разваливается, надо понимать, фильтр это эдж или срез
    выборки под удачное подмножество.
    """
    return _h(
        "oss_bb_meanrev_bare",
        {"type": "bollinger_meanrev", "params": {"period": 20, "k": 2.0}},
        {"type": "atr", "params": {"period": 14, "mult": 2.5}},
        {"type": "fixed_rr", "params": {"rr": 1.5, "max_bars": 96}},
        [],
        {"entry.params.k": [2.0, 2.5, 3.0],
         "entry.params.period": [20, 40],
         "exit.params.rr": [1.0, 1.5, 2.0]},
        "freqtrade Strategy002 (ядро) — контрольная версия без фильтра",
        "",
    )


# --------------------------------------------------------------------------- #
def phase1_queue() -> list[Hypothesis]:
    """Очередь Фазы 1: академия (№1) + классика (№3) + открытый код (№2).

    Форумы и соцсети (источник №5) — Фаза 2, подключаются последними, как
    самый шумный источник. Собственные сетапы трейдера (order block, FVG,
    дивергенции) в очередь НЕ входят: по р.6.5 они служат калибровочным
    эталоном, а не сырьём для поиска.
    """
    return [
        tsmom(), tsmom_htf_filter(),
        donchian_turtle(), keltner_vol_breakout(), ma_cross_trend(),
        bb_rsi_meanrev(), bb_meanrev_no_filter(), rsi2_connors(),
    ]


def dedup(hypos: list[Hypothesis]) -> list[Hypothesis]:
    """Отсев дубликатов по отпечатку логики (р.6.5, критерий «б»)."""
    seen, out = set(), []
    for h in hypos:
        fp = h.config.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        out.append(h)
    return out
