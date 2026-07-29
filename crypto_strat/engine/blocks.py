"""
Библиотека блоков (р.6): входы, фильтры, стопы, выходы.

Контракт входного блока — вернуть список НАМЕРЕНИЙ (OrderIntent), где
`signal_bar` = бар, на ЗАКРЫТИИ которого сигнал стал известен. Исполнитель
(backtest.py) физически не может тронуть заявку раньше signal_bar+1.
Это второй рубеж защиты от lookahead: даже если блок ошибётся, войти
«в тот же бар» невозможно by design.

Стоп задаётся не числом, а СПЕЦИФИКАЦИЕЙ: для рыночного входа цена входа
неизвестна в момент сигнала, поэтому стоп считается в момент заполнения —
но исключительно по данным, доступным на signal_bar (например ATR[signal_bar]).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


# --------------------------------------------------------------------------- #
# НАМЕРЕНИЕ НА ВХОД
# --------------------------------------------------------------------------- #
@dataclass
class OrderIntent:
    direction: int                 # +1 long, -1 short
    signal_bar: int                # бар, на закрытии которого возник сигнал
    kind: str                      # 'market' | 'limit' | 'stop'
    price: float | None            # цена лимитной/стоп-заявки; None для market
    stop_spec: dict                # как посчитать защитный стоп в момент входа
    valid_bars: int = 1            # сколько баров заявка живёт (market -> 1)
    invalidate: dict = field(default_factory=dict)   # условие снятия заявки
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# ВХОДНЫЕ БЛОКИ
# --------------------------------------------------------------------------- #
def _arrays(df: pd.DataFrame):
    return (df["open"].to_numpy(), df["high"].to_numpy(),
            df["low"].to_numpy(), df["close"].to_numpy())


def entry_order_block(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """SMC order block через BOS — методология трейдера (р.6.5 п.4).

    BOS = закрытие за подтверждённым свингом. OB = последняя противоположная
    свеча перед импульсом. Вход — лимитом при возврате в зону, стоп за зоной.
    Зона существует только с бара BOS: раньше её не существовало и знать о ней
    было нельзя.
    """
    o, h, l, c = _arrays(df)
    n = len(df)
    left = int(p.get("swing_left", 3))
    right = int(p.get("swing_right", 3))
    use_body = bool(p.get("ob_use_body", False))
    max_age = int(p.get("ob_max_age", 60))
    buf = float(p.get("stop_buffer", 0.0005))

    sh_idx, sl_idx = ind.find_swings(h, l, left, right)
    last_sh, last_sl = ind.last_confirmed_swings(n, sh_idx, sl_idx, right)

    intents: list[OrderIntent] = []
    used_bull, used_bear = set(), set()

    for j in range(1, n):
        sh = last_sh[j]
        if sh != -1 and c[j] > h[sh] and sh not in used_bull:
            ob = next((k for k in range(j, sh, -1) if c[k] < o[k]), None)
            if ob is not None:
                top = max(o[ob], c[ob]) if use_body else h[ob]
                bottom = min(o[ob], c[ob]) if use_body else l[ob]
                if top > bottom > 0:
                    intents.append(OrderIntent(
                        direction=+1, signal_bar=j, kind="limit", price=top,
                        stop_spec={"type": "level", "price": bottom * (1 - buf)},
                        valid_bars=max_age,
                        invalidate={"type": "close_beyond", "level": bottom, "side": "below"},
                        meta={"block": "order_block", "ob_bar": ob, "bos_bar": j},
                    ))
                    used_bull.add(sh)

        sl = last_sl[j]
        if sl != -1 and c[j] < l[sl] and sl not in used_bear:
            ob = next((k for k in range(j, sl, -1) if c[k] > o[k]), None)
            if ob is not None:
                top = max(o[ob], c[ob]) if use_body else h[ob]
                bottom = min(o[ob], c[ob]) if use_body else l[ob]
                if top > bottom > 0:
                    intents.append(OrderIntent(
                        direction=-1, signal_bar=j, kind="limit", price=bottom,
                        stop_spec={"type": "level", "price": top * (1 + buf)},
                        valid_bars=max_age,
                        invalidate={"type": "close_beyond", "level": top, "side": "above"},
                        meta={"block": "order_block", "ob_bar": ob, "bos_bar": j},
                    ))
                    used_bear.add(sl)
    return intents


def _cross_signals(c: np.ndarray, up: np.ndarray, dn: np.ndarray,
                   block_name: str) -> list[OrderIntent]:
    """Сигнал ТОЛЬКО на пересечении границы, а не на каждом баре за ней.

    Иначе «пробой» превращается в «мы просто выше канала» и генерирует сотни
    сигналов подряд, из которых движок берёт случайный первый после выхода из
    предыдущей сделки. Это не формализация правила, а лотерея.
    """
    intents = []
    for i in range(1, len(c)):
        if np.isnan(up[i]) or np.isnan(up[i - 1]):
            continue
        if c[i] > up[i] and c[i - 1] <= up[i - 1]:
            intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": block_name, "level": float(up[i])}))
        elif c[i] < dn[i] and c[i - 1] >= dn[i - 1]:
            intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": block_name, "level": float(dn[i])}))
    return intents


def entry_donchian_breakout(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Пробой канала Дончиана. Границы канала берутся ИСКЛЮЧАЯ текущий бар,
    иначе «пробой» тавтологичен."""
    _, h, l, c = _arrays(df)
    up, dn = ind.donchian(h, l, int(p.get("period", 20)), exclude_current=True)
    return _cross_signals(c, up, dn, "donchian_breakout")


def entry_keltner_breakout(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Пробой канала Кельтнера (волатильностный пробой)."""
    _, h, l, c = _arrays(df)
    up, _, dn = ind.keltner(h, l, c, int(p.get("ema_period", 20)),
                            int(p.get("atr_period", 14)),
                            float(p.get("mult", 2.0)), exclude_current=True)
    return _cross_signals(c, up, dn, "keltner_breakout")


def entry_tsmom(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Time-series momentum (Moskowitz/Ooi/Pedersen), адаптация под крипту.

    Сигнал = знак доходности за lookback баров. Сделка открывается только
    при СМЕНЕ знака, иначе на каждом баре плодился бы дубликат.
    """
    _, _, _, c = _arrays(df)
    lb = int(p.get("lookback", 168))
    intents, prev = [], 0
    for i in range(lb, len(df)):
        sig = 1 if c[i] > c[i - lb] else (-1 if c[i] < c[i - lb] else 0)
        if sig != 0 and sig != prev:
            intents.append(OrderIntent(sig, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "tsmom"}))
        prev = sig
    return intents


def entry_ma_cross(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Пересечение двух скользящих — базовое трендследование."""
    _, _, _, c = _arrays(df)
    fast = ind.ema(c, int(p.get("fast", 20)))
    slow = ind.ema(c, int(p.get("slow", 50)))
    intents = []
    for i in range(1, len(df)):
        if np.isnan(slow[i]) or np.isnan(slow[i - 1]):
            continue
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "ma_cross"}))
        elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "ma_cross"}))
    return intents


def entry_bollinger_meanrev(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Возврат к среднему: закрытие за полосой Боллинджера -> вход против движения."""
    _, _, _, c = _arrays(df)
    period = int(p.get("period", 20))
    k = float(p.get("k", 2.0))
    s = pd.Series(c)
    mid = s.rolling(period, min_periods=period).mean().to_numpy()
    sd = s.rolling(period, min_periods=period).std(ddof=0).to_numpy()
    up, dn = mid + k * sd, mid - k * sd
    # вход против движения -> сигнал на пересечении полосы наружу
    intents = []
    for i in range(1, len(df)):
        if np.isnan(up[i]) or np.isnan(up[i - 1]):
            continue
        if c[i] < dn[i] and c[i - 1] >= dn[i - 1]:
            intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "bollinger_meanrev"}))
        elif c[i] > up[i] and c[i - 1] <= up[i - 1]:
            intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "bollinger_meanrev"}))
    return intents


def entry_fvg(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Fair Value Gap: трёхсвечный разрыв. Вход лимитом при возврате в разрыв.

    Бычий FVG на баре i: low[i] > high[i-2] (тело импульса не перекрыло разрыв).
    Разрыв виден только на закрытии i, значит вход не раньше i+1.
    """
    o, h, l, c = _arrays(df)
    min_size = float(p.get("min_size_pct", 0.001))
    max_age = int(p.get("max_age", 40))
    buf = float(p.get("stop_buffer", 0.0005))
    intents = []
    for i in range(2, len(df)):
        # бычий разрыв
        if l[i] > h[i - 2]:
            top, bottom = l[i], h[i - 2]
            if (top - bottom) / bottom >= min_size:
                intents.append(OrderIntent(
                    +1, i, "limit", top,
                    {"type": "level", "price": bottom * (1 - buf)}, max_age,
                    invalidate={"type": "close_beyond", "level": bottom, "side": "below"},
                    meta={"block": "fvg"}))
        # медвежий разрыв
        if h[i] < l[i - 2]:
            top, bottom = l[i - 2], h[i]
            if (top - bottom) / bottom >= min_size:
                intents.append(OrderIntent(
                    -1, i, "limit", bottom,
                    {"type": "level", "price": top * (1 + buf)}, max_age,
                    invalidate={"type": "close_beyond", "level": top, "side": "above"},
                    meta={"block": "fvg"}))
    return intents


def entry_level_retest(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Пробой подтверждённого свинг-уровня и вход лимитом на его РЕТЕСТЕ."""
    o, h, l, c = _arrays(df)
    n = len(df)
    left = int(p.get("swing_left", 3))
    right = int(p.get("swing_right", 3))
    max_age = int(p.get("max_age", 30))
    stop_atr_mult = float(p.get("stop_atr_mult", 1.5))
    a = ind.atr(h, l, c, int(p.get("atr_period", 14)))
    sh_idx, sl_idx = ind.find_swings(h, l, left, right)
    last_sh, last_sl = ind.last_confirmed_swings(n, sh_idx, sl_idx, right)

    intents, used_h, used_l = [], set(), set()
    for i in range(n):
        if np.isnan(a[i]):
            continue
        sh = last_sh[i]
        if sh != -1 and c[i] > h[sh] and sh not in used_h:
            lvl = h[sh]
            intents.append(OrderIntent(
                +1, i, "limit", lvl,
                {"type": "atr", "mult": stop_atr_mult, "atr": float(a[i])}, max_age,
                invalidate={"type": "close_beyond", "level": lvl * 0.99, "side": "below"},
                meta={"block": "level_retest"}))
            used_h.add(sh)
        sl = last_sl[i]
        if sl != -1 and c[i] < l[sl] and sl not in used_l:
            lvl = l[sl]
            intents.append(OrderIntent(
                -1, i, "limit", lvl,
                {"type": "atr", "mult": stop_atr_mult, "atr": float(a[i])}, max_age,
                invalidate={"type": "close_beyond", "level": lvl * 1.01, "side": "above"},
                meta={"block": "level_retest"}))
            used_l.add(sl)
    return intents


def entry_rsi_divergence(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Дивергенция RSI на ПОДТВЕРЖДЁННЫХ свингах.

    Бычья: цена дала более низкий лоу, RSI — более высокий. Сигнал возникает
    в момент подтверждения второго свинга (свинг i виден только на i+right).
    """
    o, h, l, c = _arrays(df)
    left = int(p.get("swing_left", 3))
    right = int(p.get("swing_right", 3))
    rsi_p = int(p.get("rsi_period", 14))
    max_gap = int(p.get("max_gap", 40))
    r = ind.rsi(c, rsi_p)
    sh_idx, sl_idx = ind.find_swings(h, l, left, right)

    intents = []
    for a_i, b_i in zip(sl_idx, sl_idx[1:]):
        if b_i - a_i > max_gap or np.isnan(r[a_i]) or np.isnan(r[b_i]):
            continue
        if l[b_i] < l[a_i] and r[b_i] > r[a_i]:
            sig = b_i + right                      # момент подтверждения свинга
            if sig < len(df):
                intents.append(OrderIntent(+1, sig, "market", None,
                                           {"type": "none"}, 1,
                                           meta={"block": "rsi_divergence"}))
    for a_i, b_i in zip(sh_idx, sh_idx[1:]):
        if b_i - a_i > max_gap or np.isnan(r[a_i]) or np.isnan(r[b_i]):
            continue
        if h[b_i] > h[a_i] and r[b_i] < r[a_i]:
            sig = b_i + right
            if sig < len(df):
                intents.append(OrderIntent(-1, sig, "market", None,
                                           {"type": "none"}, 1,
                                           meta={"block": "rsi_divergence"}))
    return sorted(intents, key=lambda x: x.signal_bar)


def entry_squeeze_breakout(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Сжатие волатильности -> расширение (Bollinger squeeze).

    Механизм структурно ДРУГОЙ, чем у пробоя уровня. Пробой Дончиана говорит
    «цена вышла за экстремум». Здесь условие на РЕЖИМ ВОЛАТИЛЬНОСТИ: полосы
    Боллинджера сжались до нижнего перцентиля собственного диапазона, то есть
    рынок «сжал пружину». Вход — на первом закрытии за полосой после сжатия.

    Формализация по описанию источников: ширина полос в нижних N% своего
    диапазона за lookback баров И ATR ниже своей средней; сигнал — закрытие
    за полосой.
    """
    _, h, l, c = _arrays(df)
    n = len(df)
    period = int(p.get("period", 20))
    k = float(p.get("k", 2.0))
    lookback = int(p.get("squeeze_lookback", 120))
    pct = float(p.get("squeeze_pct", 0.30))

    s = pd.Series(c)
    mid = s.rolling(period, min_periods=period).mean()
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    width = (2 * k * sd / mid).to_numpy()          # относительная ширина полос
    up = (mid + k * sd).to_numpy()
    dn = (mid - k * sd).to_numpy()

    # порог сжатия — нижний перцентиль ширины за lookback, БЕЗ текущего бара
    thr = pd.Series(width).shift(1).rolling(lookback, min_periods=lookback) \
            .quantile(pct).to_numpy()
    a = ind.atr(h, l, c, int(p.get("atr_period", 14)))
    a_ma = ind.sma(a, int(p.get("atr_ma", 20)))

    intents = []
    for i in range(1, n):
        if np.isnan(thr[i]) or np.isnan(width[i]) or np.isnan(a_ma[i]):
            continue
        squeezed = width[i] <= thr[i] and a[i] <= a_ma[i]
        if not squeezed:
            continue
        if c[i] > up[i] and c[i - 1] <= up[i - 1]:
            intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "squeeze_breakout"}))
        elif c[i] < dn[i] and c[i - 1] >= dn[i - 1]:
            intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "squeeze_breakout"}))
    return intents


def entry_ttm_squeeze(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """TTM Squeeze: полосы Боллинджера ВНУТРИ канала Кельтнера = сжатие.

    Сигнал не на самом сжатии, а на его ОТПУСКАНИИ (полосы выходят за Кельтнер),
    направление берётся по знаку моментума. Отличие от `squeeze_breakout`:
    там сжатие меряется перцентилем собственной ширины, здесь — сравнением
    двух каналов разной природы (стандартное отклонение против ATR).
    """
    _, h, l, c = _arrays(df)
    n = len(df)
    period = int(p.get("period", 20))
    k = float(p.get("k", 2.0))
    mult = float(p.get("kc_mult", 1.5))
    mom_p = int(p.get("mom_period", 12))

    s = pd.Series(c)
    mid = s.rolling(period, min_periods=period).mean().to_numpy()
    sd = s.rolling(period, min_periods=period).std(ddof=0).to_numpy()
    bb_up, bb_dn = mid + k * sd, mid - k * sd
    kc_up, _, kc_dn = ind.keltner(h, l, c, period, int(p.get("atr_period", 14)),
                                  mult, exclude_current=False)

    on = (bb_up < kc_up) & (bb_dn > kc_dn)          # сжатие включено
    intents = []
    for i in range(mom_p + 1, n):
        if np.isnan(bb_up[i]) or np.isnan(kc_up[i]) or np.isnan(kc_up[i - 1]):
            continue
        # отпускание: на предыдущем баре сжатие было, на текущем — нет
        if on[i - 1] and not on[i]:
            mom = c[i] - c[i - mom_p]
            if mom > 0:
                intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                           meta={"block": "ttm_squeeze"}))
            elif mom < 0:
                intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                           meta={"block": "ttm_squeeze"}))
    return intents


def entry_nr_expansion(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Самый узкий бар за N периодов -> пробой его диапазона (схема NR7).

    Ещё один срез той же идеи, но без индикаторов вообще: узкий бар = пауза,
    выход за его границы = возобновление движения. Заявки стоп-типа по обе
    стороны бара, живут ограниченное число баров.
    """
    _, h, l, c = _arrays(df)
    n = len(df)
    N = int(p.get("lookback", 7))
    valid = int(p.get("valid_bars", 3))
    rng = h - l
    intents = []
    for i in range(N, n):
        window = rng[i - N + 1:i + 1]
        if rng[i] != window.min() or (window == rng[i]).sum() != 1 or rng[i] <= 0:
            continue
        intents.append(OrderIntent(+1, i, "stop", float(h[i]), {"type": "none"},
                                   valid, meta={"block": "nr_expansion"}))
        intents.append(OrderIntent(-1, i, "stop", float(l[i]), {"type": "none"},
                                   valid, meta={"block": "nr_expansion"}))
    return intents


def entry_rsi_threshold(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Вход по пересечению порога RSI (возврат к среднему, схема Connors RSI-2).

    Сигнал только на ПЕРЕСЕЧЕНИИ порога вниз/вверх: удержание RSI под уровнем
    десять баров подряд — это одна ситуация, а не десять сигналов.
    """
    _, _, _, c = _arrays(df)
    r = ind.rsi(c, int(p.get("period", 2)))
    lo = float(p.get("oversold", 5.0))
    hi = float(p.get("overbought", 95.0))
    intents = []
    for i in range(1, len(df)):
        if np.isnan(r[i]) or np.isnan(r[i - 1]):
            continue
        if r[i] < lo <= r[i - 1]:
            intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "rsi_threshold"}))
        elif r[i] > hi >= r[i - 1]:
            intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "rsi_threshold"}))
    return intents


# --------------------------------------------------------------------------- #
# МЕЖРЫНОЧНЫЕ БЛОКИ — читают ВТОРОЙ инструмент (ведущий)
# --------------------------------------------------------------------------- #
class MissingReferenceError(RuntimeError):
    """Межрыночный блок запрошен без данных ведущего инструмента."""


def _ref_close(df: pd.DataFrame, ref_name: str) -> np.ndarray:
    """Закрытия ведущего инструмента, выровненные на бары текущего.

    Выравнивание по ТАЙМСТАМПУ, а не по позиции: у инструментов разная длина
    истории (SOL с 2020, BTC с 2017), и позиционное совмещение сдвинуло бы
    ряды на годы.

    Причинность: берётся последний бар ведущего с ts <= ts текущего бара.
    Оба инструмента на одном ТФ закрываются одновременно, поэтому значение
    известно на закрытии текущего бара — а вход всё равно не раньше следующего.
    """
    refs = df.attrs.get("refs") or {}
    ref = refs.get(ref_name)
    if ref is None or not len(ref):
        raise MissingReferenceError(
            f"нет данных ведущего инструмента {ref_name} — межрыночный блок "
            f"невозможен")
    r_ts = ref["ts"].to_numpy()
    r_c = ref["close"].to_numpy()
    pos = np.searchsorted(r_ts, df["ts"].to_numpy(), side="right") - 1
    out = np.full(len(df), np.nan)
    ok = pos >= 0
    out[ok] = r_c[pos[ok]]
    return out


def entry_ref_momentum(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Импульс ВЕДУЩЕГО инструмента как сигнал на ведомом (спилловер).

    Гипотеза: информация распространяется по рынку с задержкой из-за
    ограниченного внимания инвесторов, поэтому движение BTC предсказывает
    движение альтов в ТУ ЖЕ сторону. Сигнал на пересечении порога, чтобы
    одно событие не порождало серию входов.
    """
    ref_name = p.get("ref", "BTCUSDT")
    lb = int(p.get("lookback", 6))
    thr = float(p.get("threshold", 0.03))
    sign = -1 if bool(p.get("seesaw", False)) else 1
    rc = _ref_close(df, ref_name)
    block = "ref_seesaw" if sign < 0 else "ref_momentum"

    intents, prev = [], 0
    for i in range(lb, len(df)):
        if np.isnan(rc[i]) or np.isnan(rc[i - lb]) or rc[i - lb] <= 0:
            continue
        ret = rc[i] / rc[i - lb] - 1.0
        cur = 1 if ret > thr else (-1 if ret < -thr else 0)
        if cur != 0 and cur != prev:
            intents.append(OrderIntent(sign * cur, i, "market", None,
                                       {"type": "none"}, 1,
                                       meta={"block": block, "ref": ref_name,
                                             "ref_ret": float(ret)}))
        prev = cur
    return intents


def entry_ref_seesaw(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """«Качели» (SSRN 3465924): крупные монеты предсказывают альты В МИНУС.

    Прямо противоположно народному «BTC растёт — альты растут». Академия
    объясняет это перетоком внимания: капитал бежит В крупные монеты и ИЗ них,
    а не разливается равномерно. Проверяем обе версии — какая из них верна,
    решают барьеры, а не то, какая приятнее звучит.
    """
    return entry_ref_momentum(df, {**p, "seesaw": True})


def filter_ref_trend(df: pd.DataFrame, p: dict):
    """Фильтр по тренду ведущего инструмента: торгуем альт только по
    направлению BTC. Не сигнал, а разрешение — комбинируется с любым входом."""
    ref_name = p.get("ref", "BTCUSDT")
    rc = _ref_close(df, ref_name)
    e = ind.ema(rc[~np.isnan(rc)], int(p.get("ema_period", 50)))
    full = np.full(len(df), np.nan)
    full[~np.isnan(rc)] = e
    up = (~np.isnan(full)) & (rc > full)
    dn = (~np.isnan(full)) & (rc < full)
    return up, dn


def conqueror_factors(df: pd.DataFrame, p: dict):
    """Три фактора Conqueror (Courtney Smith, гл. 4) и их знаки.

    Cond1 = close − SMA10(close)          — цена относительно средней
    Cond2 = SMA10(t) − SMA10(t−10)        — наклон самой средней
    Cond3 = close − close[−40]            — долгосрочный моментум

    Все три > 0 -> лонг, все три < 0 -> шорт, иначе вне рынка.

    Функция вынесена отдельно, потому что нужна ДВАЖДЫ: входному блоку — чтобы
    поймать момент согласия, и выходу — чтобы считать смены знака. Считать их
    в двух местах по-разному было бы источником расхождений.
    """
    _, _, _, c = _arrays(df)
    sma_p = int(p.get("sma_period", 10))
    slope_lb = int(p.get("slope_lookback", 10))
    mom_lb = int(p.get("mom_lookback", 40))

    sma = ind.sma(c, sma_p)
    f1 = c - sma
    f2 = np.full(len(c), np.nan)
    f2[slope_lb:] = sma[slope_lb:] - sma[:-slope_lb]
    f3 = np.full(len(c), np.nan)
    f3[mom_lb:] = c[mom_lb:] - c[:-mom_lb]

    signs = np.zeros((3, len(c)), dtype=np.int8)
    for i, f in enumerate((f1, f2, f3)):
        s = np.sign(np.nan_to_num(f, nan=0.0)).astype(np.int8)
        s[np.isnan(f)] = 0
        signs[i] = s

    valid = ~(np.isnan(f1) | np.isnan(f2) | np.isnan(f3))
    state = np.zeros(len(c), dtype=np.int8)
    state[valid & (signs[0] > 0) & (signs[1] > 0) & (signs[2] > 0)] = 1
    state[valid & (signs[0] < 0) & (signs[1] < 0) & (signs[2] < 0)] = -1

    # смена знака ЛЮБОГО из трёх факторов на баре k (оба знака ненулевые)
    changes = np.zeros(len(c), dtype=np.int8)
    for i in range(3):
        s = signs[i]
        flip = np.zeros(len(c), dtype=bool)
        flip[1:] = (s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0)
        changes += flip.astype(np.int8)
    return state, changes, signs


def entry_conqueror(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Вход Conqueror: сигнал на баре, где выполнилось ПОСЛЕДНЕЕ из трёх условий.

    То есть в момент ПЕРЕХОДА состояния в «все три согласны». Пока согласие
    держится, новых входов нет — иначе одно событие плодило бы серию заявок.

    Оговорка об исполнении. Первоисточник говорит «вход по закрытию бара».
    Движок исполняет рыночную заявку по ОТКРЫТИЮ следующего бара — это
    сознательно консервативнее. В крипте торговля непрерывная и 24/7, поэтому
    открытие следующего бара практически совпадает с закрытием текущего, а
    правило «не входить на баре, который только что увидел» — наш главный
    рубеж против lookahead, и ослаблять его ради буквы источника нельзя.
    """
    state, _, _ = conqueror_factors(df, p)
    intents = []
    for i in range(1, len(df)):
        if state[i] != 0 and state[i] != state[i - 1]:
            intents.append(OrderIntent(int(state[i]), i, "market", None,
                                       {"type": "none"}, 1,
                                       meta={"block": "conqueror"}))
    return intents


def entry_consensus(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """СОГЛАСИЕ нескольких механизмов разной природы.

    Вход только когда независимые сигналы совпали по направлению в пределах
    окна `window` баров. Это НЕ «навесить фильтров, пока метрика не улучшится»:
    компоненты — полноценные самостоятельные механизмы, каждый со своей
    рыночной логикой, и каждый проверен поодиночке в предыдущих прогонах.

    Гипотеза комбинирования: одиночный сигнал несёт и эдж, и шум; шум у
    механизмов РАЗНОЙ природы некоррелирован, а эдж — если он общий — нет.
    Тогда пересечение режет шум сильнее, чем эдж.

    Причинность: компонент сигналит на закрытии своего бара, согласие
    фиксируется на баре последнего подтвердившего, вход — не раньше следующего.
    Ни один компонент не может «дождаться» будущего.

    Сигнал возникает только в момент ВОЗНИКНОВЕНИЯ согласия. Пока согласие
    держится, новых входов нет — иначе одно событие плодило бы серию заявок.
    """
    comps = p.get("components") or []
    if len(comps) < 2:
        raise ValueError("consensus требует минимум 2 компонента")
    window = int(p.get("window", 6))
    need = int(p.get("min_agree", len(comps)))
    n = len(df)

    # для каждого компонента — бар последнего сигнала по каждому направлению
    last_sig = np.full((len(comps), 2, n), -10**9, dtype=np.int64)   # [comp][dir][bar]
    for ci, spec in enumerate(comps):
        fn = ENTRY_BLOCKS.get(spec["type"])
        if fn is None:
            raise ValueError(f"неизвестный компонент согласия: {spec['type']}")
        for di in (0, 1):
            last_sig[ci, di, :] = -10**9
        cur = [-10**9, -10**9]
        marks = {0: [], 1: []}
        for it in fn(df, spec.get("params", {})):
            marks[0 if it.direction > 0 else 1].append(it.signal_bar)
        for di in (0, 1):
            arr = np.full(n, -10**9, dtype=np.int64)
            last = -10**9
            ptr, ms = 0, sorted(marks[di])
            for i in range(n):
                while ptr < len(ms) and ms[ptr] <= i:
                    last = ms[ptr]; ptr += 1
                arr[i] = last
            last_sig[ci, di, :] = arr

    intents, active = [], 0
    for i in range(n):
        fired = 0
        for di, d in ((0, +1), (1, -1)):
            agree = int((i - last_sig[:, di, i] < window).sum())
            if agree >= need:
                if active != d:
                    intents.append(OrderIntent(
                        d, i, "market", None, {"type": "none"}, 1,
                        meta={"block": "consensus", "agree": agree,
                              "components": [c["type"] for c in comps]}))
                    active = d
                fired = d
                break
        if not fired:
            active = 0
    return sorted(intents, key=lambda x: x.signal_bar)


# --------------------------------------------------------------------------- #
# БЛОКИ КУРТНИ СМИТА — извлечены из первоисточника (книга), не из чужих скриптов
#
# Общая оговорка об ИСПОЛНЕНИИ, действующая на все блоки этого раздела.
# Смит многократно пишет «входим/выходим по цене закрытия». Движок исполняет
# такие приказы по ОТКРЫТИЮ СЛЕДУЮЩЕГО бара — сознательно консервативнее.
# Причина в том, что «решение принято по закрытию, исполнено по этому же
# закрытию» — это ровно тот шов, где lookahead и заводится. Автор, кстати,
# сам разрешает такую замену: «если вы не можете выйти по цене закрытия...
# то вам остается выйти при первой же возможности».
# Стоп- и лимит-приказы, наоборот, исполняются ВНУТРИ бара — они физически
# лежат у брокера заранее, никакого знания о будущем не требуют.
# --------------------------------------------------------------------------- #
def entry_channel_stop(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Прорыв канала СТОП-ПРИКАЗАМИ (Смит, гл. 3) — не по закрытию.

    Отличие от `donchian_breakout` принципиальное, а не косметическое.
    Дончиан в нашей библиотеке даёт сигнал на ЗАКРЫТИИ бара за границей канала,
    то есть вход происходит уже после того, как весь импульс бара отработан.
    Смит же держит у брокера приказ чуть выше N-дневного максимума и чуть ниже
    N-дневного минимума и переставляет их КАЖДЫЙ день. Исполнение — внутри
    бара, по цене уровня: «Я предпочитаю покупать на уровне три пипса выше
    уровня прорыва».

    Причинность: максимум окна [i-N+1, i] известен на закрытии бара i, а приказ
    выставляется на бар i+1. Заглянуть вперёд негде.

    Правило отсечения (тоже гл. 3) навешивается флагом `cutoff`: в мету заявки
    кладётся уровень прорыва и признак «условие пяти дней выполнено» —
    выход ими пользуется, сам вход не меняется.
    """
    _, h, l, c = _arrays(df)
    n = len(df)
    period = int(p.get("period", 55))
    buf = float(p.get("buffer", 0.0002))
    cutoff = bool(p.get("cutoff", False))
    flat_days = int(p.get("flat_days", 5))

    up = pd.Series(h).rolling(period, min_periods=period).max().to_numpy()
    dn = pd.Series(l).rolling(period, min_periods=period).min().to_numpy()

    def _flat(arr, i, rising: bool) -> bool:
        """Условие пяти дней: граница канала стояла на месте или шла ПРОТИВ
        прорыва как минимум flat_days баров. Смысл авторский: отсечение
        применяется только к боковому рынку, а не к сильному тренду."""
        if i - flat_days < 0:
            return False
        for j in range(flat_days):
            a, b = arr[i - j], arr[i - j - 1]
            if np.isnan(a) or np.isnan(b):
                return False
            if (a > b) if rising else (a < b):
                return False
        return True

    intents = []
    for i in range(n - 1):
        if np.isnan(up[i]) or np.isnan(dn[i]):
            continue
        lvl_u, lvl_d = float(up[i]), float(dn[i])
        m_u = {"block": "channel_stop", "level": lvl_u}
        m_d = {"block": "channel_stop", "level": lvl_d}
        if cutoff:
            m_u["cutoff"] = _flat(up, i, rising=True)
            m_d["cutoff"] = _flat(dn, i, rising=False)
        intents.append(OrderIntent(+1, i, "stop", lvl_u * (1 + buf),
                                   {"type": "none"}, 1, meta=m_u))
        intents.append(OrderIntent(-1, i, "stop", lvl_d * (1 - buf),
                                   {"type": "none"}, 1, meta=m_d))
    return intents


def smith_swing_state(df: pd.DataFrame, p: dict):
    """Свинги по Демарку в трактовке Смита и структура тренда (гл. 2).

    Определение свинг-хая у Смита: справа бар с более низким максимумом,
    слева — не менее N баров с более низкими максимумами. Ранг = сколько баров
    слева; значимыми автор считает ТРЁХБАРНЫЕ, одно- и двухбарные игнорирует.
    Отсюда left=3, right=1.

    Структура: бычий рынок = растут и максимумы, и минимумы; медвежий = падают
    и те, и другие; всё остальное — нейтральный. Свинг становится известен
    только на баре подтверждения (i+right) — до этого момента система о нём
    не знает и знать не может.

    Возвращает по барам: (state, hi_dir, lo_dir, hi_lvl, lo_lvl, n_hi, n_lo),
    где hi_dir/lo_dir — направление ДВУХ последних свингов соответствующего
    типа, а n_hi/n_lo — сколько свингов каждого типа УЖЕ подтверждено к бару.
    Счётчики нужны выходу: он обязан отличать «структура изменилась» от
    «структура ещё не успела догнать наш пробой».
    """
    _, h, l, _c = _arrays(df)
    n = len(df)
    left = int(p.get("swing_left", 3))
    right = int(p.get("swing_right", 1))
    sh, sl = ind.find_swings(h, l, left, right)

    state = np.zeros(n, dtype=np.int8)
    hi_dir = np.zeros(n, dtype=np.int8)
    lo_dir = np.zeros(n, dtype=np.int8)
    hi_lvl = np.full(n, np.nan)
    lo_lvl = np.full(n, np.nan)
    n_hi = np.zeros(n, dtype=np.int32)
    n_lo = np.zeros(n, dtype=np.int32)

    ph = pl = 0
    hs: list[float] = []
    ls: list[float] = []
    for j in range(n):
        while ph < len(sh) and sh[ph] + right <= j:
            hs.append(float(h[sh[ph]])); ph += 1
        while pl < len(sl) and sl[pl] + right <= j:
            ls.append(float(l[sl[pl]])); pl += 1
        if hs:
            hi_lvl[j] = hs[-1]
        if ls:
            lo_lvl[j] = ls[-1]
        if len(hs) >= 2:
            hi_dir[j] = 1 if hs[-1] > hs[-2] else (-1 if hs[-1] < hs[-2] else 0)
        if len(ls) >= 2:
            lo_dir[j] = 1 if ls[-1] > ls[-2] else (-1 if ls[-1] < ls[-2] else 0)
        n_hi[j], n_lo[j] = len(hs), len(ls)
        if hi_dir[j] > 0 and lo_dir[j] > 0:
            state[j] = 1
        elif hi_dir[j] < 0 and lo_dir[j] < 0:
            state[j] = -1
    return state, hi_dir, lo_dir, hi_lvl, lo_lvl, n_hi, n_lo


def entry_trend_swings(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Анализ тренда по свингам (Смит, гл. 2) — вход НА ОПЕРЕЖЕНИЕ.

    Ключевая мысль автора, которую легко потерять при формализации: он не ждёт
    завершения структуры. Увидев более низкий максимум (половина определения
    медвежьего рынка), он ставит приказ на продажу у последнего минимума
    колебания — «мне не надо ждать, пока мы достигнем более низкого минимума,
    чтобы начать играть на понижение. Я просто должен знать, что это
    произойдёт».

    Отсюда правило: hi_dir = -1 -> стоп-приказ на продажу у последнего
    свинг-лоу; lo_dir = +1 -> стоп-приказ на покупку у последнего свинг-хая.
    Само исполнение приказа И ЕСТЬ момент, когда структура становится
    направленной: пробив последний лоу при падающих хаях, рынок создаёт более
    низкий лоу, то есть медвежью структуру.

    Приказы переставляются каждый бар, потому что уровни двигаются вместе с
    появлением новых подтверждённых свингов.
    """
    _, h, l, _c = _arrays(df)
    n = len(df)
    _state, hi_dir, lo_dir, hi_lvl, lo_lvl, _nh, _nl = smith_swing_state(df, p)
    buf = float(p.get("buffer", 0.0002))

    intents = []
    for i in range(n - 1):
        if lo_dir[i] > 0 and not np.isnan(hi_lvl[i]):
            intents.append(OrderIntent(
                +1, i, "stop", float(hi_lvl[i]) * (1 + buf), {"type": "none"}, 1,
                meta={"block": "trend_swings", "level": float(hi_lvl[i])}))
        if hi_dir[i] < 0 and not np.isnan(lo_lvl[i]):
            intents.append(OrderIntent(
                -1, i, "stop", float(lo_lvl[i]) * (1 - buf), {"type": "none"}, 1,
                meta={"block": "trend_swings", "level": float(lo_lvl[i])}))
    return intents


def entry_stoch_cross50(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Стохастик как механическая система (Смит, гл. 5).

    Дословно: «покупать, когда стохастик %K пересекает линию на отметке выше
    50, и продавать, когда ниже 50. Вы всегда будете присутствовать на рынке».
    Это единственное полностью механическое правило главы — дивергенции автор
    торгует дискреционно («открою короткую, если увижу любой предлог»), и
    формализовать их без домысливания нельзя, поэтому они и не формализуются.
    """
    _, h, l, c = _arrays(df)
    k, _d = ind.stochastic(h, l, c, int(p.get("k_period", 14)),
                           int(p.get("k_smooth", 3)), int(p.get("d_period", 3)))
    intents = []
    for i in range(1, len(df)):
        if np.isnan(k[i]) or np.isnan(k[i - 1]):
            continue
        if k[i] > 50.0 >= k[i - 1]:
            intents.append(OrderIntent(+1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "stoch_cross50"}))
        elif k[i] < 50.0 <= k[i - 1]:
            intents.append(OrderIntent(-1, i, "market", None, {"type": "none"}, 1,
                                       meta={"block": "stoch_cross50"}))
    return intents


def entry_inside_day(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Внутренний день (Смит, гл. 6) — «мини-версия прорыва канала».

    Внутренний день = весь диапазон укладывается в диапазон предыдущего:
    максимум ниже, минимум выше. Автор трактует это как ничью быков и медведей
    и ставит стоп-приказы по обе стороны бара, чтобы пойти за победившей
    стороной. Защитный стоп — за противоположной границей того же бара.
    Позиция закрывается в тот же день (у нас — по открытию следующего бара,
    см. общую оговорку об исполнении выше).
    """
    _o, h, l, _c = _arrays(df)
    buf = float(p.get("buffer", 0.0002))
    sbuf = float(p.get("stop_buffer", 0.0005))
    intents = []
    for i in range(1, len(df) - 1):
        if not (h[i] < h[i - 1] and l[i] > l[i - 1]):
            continue
        if h[i] <= l[i]:
            continue
        intents.append(OrderIntent(
            +1, i, "stop", float(h[i]) * (1 + buf),
            {"type": "level", "price": float(l[i]) * (1 - sbuf)}, 1,
            meta={"block": "inside_day"}))
        intents.append(OrderIntent(
            -1, i, "stop", float(l[i]) * (1 - buf),
            {"type": "level", "price": float(h[i]) * (1 + sbuf)}, 1,
            meta={"block": "inside_day"}))
    return intents


def entry_reversal_day(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """День разворота (Смит, гл. 6).

    Бычий: бар ушёл НИЖЕ минимума предыдущего дня, но закрылся ВЫШЕ его
    закрытия — медведи начали день, быки его забрали. Медвежий — зеркально.
    Автор входит под закрытие дня разворота и держит до закрытия следующего;
    защитный стоп — за экстремумом дня разворота.
    """
    _o, h, l, c = _arrays(df)
    sbuf = float(p.get("stop_buffer", 0.0005))
    intents = []
    for i in range(1, len(df)):
        if l[i] < l[i - 1] and c[i] > c[i - 1]:
            intents.append(OrderIntent(
                +1, i, "market", None,
                {"type": "level", "price": float(l[i]) * (1 - sbuf)}, 1,
                meta={"block": "reversal_day"}))
        elif h[i] > h[i - 1] and c[i] < c[i - 1]:
            intents.append(OrderIntent(
                -1, i, "market", None,
                {"type": "level", "price": float(h[i]) * (1 + sbuf)}, 1,
                meta={"block": "reversal_day"}))
    return intents


def entry_slingshot(df: pd.DataFrame, p: dict) -> list[OrderIntent]:
    """Slingshot / Mini-Slingshot (Смит, гл. 8).

    Медвежья формация (бычья зеркальна):
      * ГЛАВНЫЙ МАКСИМУМ — свинг-хай выше предыдущего свинг-хая;
      * ГЛАВНЫЙ МИНИМУМ — свинг-лоу после него;
      * МАКСИМУМ SLINGSHOT — следующий свинг-хай, который НЕ смог обновить
        главный максимум («этот максимум — провал»).
    Два подтверждения, ради которых метод и существует:
      1) между главным максимумом и максимумом Slingshot не меньше `min_gap`
         баров — гарантия, что рынок устоялся, а не рвётся;
      2) не больше `max_gap` баров — гарантия, что мы не ловим дно.
    Вход — стоп-приказ на прорыв главного минимума. Стоп — максимум Slingshot.
    Цель прибыли: из главного минимума вычитается (макс. Slingshot − главный
    минимум), то есть проекция всей формации вниз от точки входа.

    Mini-Slingshot (`mini=True`) снимает верхнее ограничение по расстоянию и
    смягчает нижнее — автор явно жертвует качеством подтверждений ради частоты.

    Причинность: свинг известен только на баре подтверждения (i+right), поэтому
    сигнал ставится там, а не на самом экстремуме.
    """
    _o, h, l, _c = _arrays(df)
    n = len(df)
    left = int(p.get("swing_left", 2))
    right = int(p.get("swing_right", 2))
    min_gap = int(p.get("min_gap", 3))
    max_gap = int(p.get("max_gap", 20))
    mini = bool(p.get("mini", False))
    if mini:
        min_gap = int(p.get("min_gap", 2))
        max_gap = 10 ** 9
    max_age = int(p.get("max_age", 20))
    sbuf = float(p.get("stop_buffer", 0.0005))
    buf = float(p.get("buffer", 0.0002))

    sh, sl = ind.find_swings(h, l, left, right)
    sh_set = sorted(sh)
    sl_set = sorted(sl)
    intents = []

    def _between(lows, a, b):
        return [i for i in lows if a < i < b]

    # --- медвежья формация: главный максимум -> главный минимум -> Slingshot ---
    for idx in range(1, len(sh_set)):
        b = sh_set[idx]                      # кандидат в максимум Slingshot
        a = None
        for prev in reversed(sh_set[:idx]):  # последний БОЛЕЕ ВЫСОКИЙ свинг-хай
            if h[prev] > h[b]:
                a = prev
                break
        if a is None:
            continue
        gap = b - a
        if gap < min_gap or gap > max_gap:
            continue
        mids = _between(sl_set, a, b)
        if not mids:
            continue
        ml = min(mids, key=lambda i: l[i])   # главный минимум формации
        sig = b + right                      # бар подтверждения максимума Slingshot
        if sig >= n:
            continue
        lo, hi = float(l[ml]), float(h[b])
        if hi <= lo:
            continue
        intents.append(OrderIntent(
            -1, sig, "stop", lo * (1 - buf),
            {"type": "level", "price": hi * (1 + sbuf)}, max_age,
            invalidate={"type": "close_beyond", "level": hi, "side": "above"},
            meta={"block": "slingshot", "target": lo - (hi - lo),
                  "sling": hi, "major": lo}))

    # --- бычья формация (зеркально) ---
    for idx in range(1, len(sl_set)):
        b = sl_set[idx]
        a = None
        for prev in reversed(sl_set[:idx]):
            if l[prev] < l[b]:
                a = prev
                break
        if a is None:
            continue
        gap = b - a
        if gap < min_gap or gap > max_gap:
            continue
        mids = _between(sh_set, a, b)
        if not mids:
            continue
        mh = max(mids, key=lambda i: h[i])
        sig = b + right
        if sig >= n:
            continue
        hi, lo = float(h[mh]), float(l[b])
        if hi <= lo:
            continue
        intents.append(OrderIntent(
            +1, sig, "stop", hi * (1 + buf),
            {"type": "level", "price": lo * (1 - sbuf)}, max_age,
            invalidate={"type": "close_beyond", "level": lo, "side": "below"},
            meta={"block": "slingshot", "target": hi + (hi - lo),
                  "sling": lo, "major": hi}))

    return sorted(intents, key=lambda x: x.signal_bar)


ENTRY_BLOCKS = {
    "order_block": entry_order_block,
    "channel_stop": entry_channel_stop,
    "trend_swings": entry_trend_swings,
    "stoch_cross50": entry_stoch_cross50,
    "inside_day": entry_inside_day,
    "reversal_day": entry_reversal_day,
    "slingshot": entry_slingshot,
    "rsi_threshold": entry_rsi_threshold,
    "consensus": entry_consensus,
    "conqueror": entry_conqueror,
    "squeeze_breakout": entry_squeeze_breakout,
    "ttm_squeeze": entry_ttm_squeeze,
    "nr_expansion": entry_nr_expansion,
    "ref_momentum": entry_ref_momentum,
    "ref_seesaw": entry_ref_seesaw,
    "donchian_breakout": entry_donchian_breakout,
    "keltner_breakout": entry_keltner_breakout,
    "tsmom": entry_tsmom,
    "ma_cross": entry_ma_cross,
    "bollinger_meanrev": entry_bollinger_meanrev,
    "fvg": entry_fvg,
    "level_retest": entry_level_retest,
    "rsi_divergence": entry_rsi_divergence,
}


# --------------------------------------------------------------------------- #
# ФИЛЬТРЫ — возвращают (mask_long, mask_short) по барам
# --------------------------------------------------------------------------- #
def filter_htf_trend(df: pd.DataFrame, p: dict):
    """Тренд старшего ТФ. Использует ТОЛЬКО закрытые старшие бары."""
    t = ind.htf_trend_causal(df, int(p.get("factor", 4)), int(p.get("ema_period", 50)))
    return t > 0, t < 0


def filter_session(df: pd.DataFrame, p: dict):
    """Час суток UTC в диапазоне [start, end). Крипта 24/7, но ликвидность
    и характер движения по сессиям различаются."""
    hours = df["dt_utc"].dt.hour.to_numpy()
    s, e = int(p.get("start_hour", 6)), int(p.get("end_hour", 20))
    m = (hours >= s) & (hours < e) if s < e else (hours >= s) | (hours < e)
    return m, m


def filter_atr_regime(df: pd.DataFrame, p: dict):
    """Режим волатильности: ATR/close в коридоре. Отсекает и мёртвый флэт,
    и экстремальный шум, где издержки и проскальзывание съедают всё."""
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    a = ind.atr(h, l, c, int(p.get("period", 14))) / c
    lo, hi = float(p.get("min_pct", 0.0)), float(p.get("max_pct", 1.0))
    m = (~np.isnan(a)) & (a >= lo) & (a <= hi)
    return m, m


class MissingVolumeError(RuntimeError):
    """Объёмный блок запрошен на данных без объёма."""


def has_volume(df: pd.DataFrame) -> bool:
    if "has_volume" in df.attrs:
        return bool(df.attrs["has_volume"])
    v = df.get("volume")
    return v is not None and bool(v.notna().any() and (v.fillna(0) > 0).any())


def filter_volume(df: pd.DataFrame, p: dict):
    """Объём выше своей средней — подтверждение интереса.

    Если объёма в данных нет, блок ПАДАЕТ, а не работает на нулях. Молча
    вернуть маску из NaN-сравнений было бы хуже всего: фильтр либо пропустил бы
    всё, либо не пропустил ничего, и в обоих случаях результат выглядел бы
    осмысленным. Лучше явный отказ — гипотеза просто не попадёт в очередь.
    """
    if not has_volume(df):
        raise MissingVolumeError(
            "объёмный фильтр невозможен: в данных нет объёма "
            "(H4-котировки идут без volume)")
    v = df["volume"].to_numpy()
    ma = ind.sma(v, int(p.get("period", 20)))
    m = (~np.isnan(ma)) & (v > ma * float(p.get("mult", 1.0)))
    return m, m


def filter_rsi_bound(df: pd.DataFrame, p: dict):
    """RSI-коридор: лонги не берём в перекупленности, шорты — в перепроданности."""
    c = df["close"].to_numpy()
    r = ind.rsi(c, int(p.get("period", 14)))
    lo, hi = float(p.get("min", 0.0)), float(p.get("max", 100.0))
    ok = (~np.isnan(r)) & (r >= lo) & (r <= hi)
    return ok, ok


def filter_ema_slope(df: pd.DataFrame, p: dict):
    """Наклон EMA — грубый, но честный признак направленности рынка."""
    c = df["close"].to_numpy()
    e = ind.ema(c, int(p.get("period", 100)))
    lag = int(p.get("lag", 20))
    slope = np.full(len(c), np.nan)
    slope[lag:] = (e[lag:] - e[:-lag]) / np.where(e[:-lag] == 0, np.nan, e[:-lag])
    thr = float(p.get("min_slope", 0.0))
    return slope > thr, slope < -thr


def filter_ma_side(df: pd.DataFrame, p: dict):
    """Цена выше/ниже длинной скользящей — классический трендовый фильтр
    (у Connors это SMA-200: лонги только выше неё, шорты только ниже)."""
    c = df["close"].to_numpy()
    ma = (ind.sma(c, int(p.get("period", 200))) if p.get("kind", "sma") == "sma"
          else ind.ema(c, int(p.get("period", 200))))
    ok = ~np.isnan(ma)
    return ok & (c > ma), ok & (c < ma)


def filter_squeeze(df: pd.DataFrame, p: dict):
    """Режим волатильности как УСЛОВИЕ: рынок сжат (или, наоборот, разжат).

    Тот же измеритель, что в `squeeze_breakout`, но в роли разрешения, а не
    сигнала. Нужно для ансамблей: «пробой засчитываем только если он случился
    из сжатия» — это пересечение двух независимых механизмов, а не два подряд
    приложенных фильтра одной природы.
    """
    _, h, l, c = _arrays(df)
    period = int(p.get("period", 20))
    k = float(p.get("k", 2.0))
    lookback = int(p.get("lookback", 120))
    pct = float(p.get("pct", 0.30))
    want_squeezed = bool(p.get("squeezed", True))

    s = pd.Series(c)
    mid = s.rolling(period, min_periods=period).mean()
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    width = (2 * k * sd / mid).to_numpy()
    thr = pd.Series(width).shift(1).rolling(lookback, min_periods=lookback) \
            .quantile(pct).to_numpy()
    ok = ~(np.isnan(width) | np.isnan(thr))
    m = ok & ((width <= thr) if want_squeezed else (width > thr))
    return m, m


def filter_adx_rising(df: pd.DataFrame, p: dict):
    """Фильтр ADX Смита (гл. 2 и 3): берём сигнал, только если ADX ВЫШЕ, чем
    вчера. «Не обращайте внимания на поступающие сигналы, если ADX ниже, чем
    был день назад».

    Логика автора: и анализ тренда, и прорывы канала — техники следования за
    трендом, они зарабатывают на сильных движениях и теряют в их отсутствие.
    Растущий ADX означает, что движение усиливается; падающий — что рынок
    сваливается в боковик, где трендовая техника обречена платить издержки.
    Направление ADX не различает лонг и шорт, поэтому маска общая.
    """
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    a, _pdi, _mdi = ind.adx(h, l, c, int(p.get("period", 14)))
    m = np.zeros(len(df), dtype=bool)
    m[1:] = (~np.isnan(a[1:])) & (~np.isnan(a[:-1])) & (a[1:] > a[:-1])
    return m, m


FILTER_BLOCKS = {
    "htf_trend": filter_htf_trend,
    "adx_rising": filter_adx_rising,
    "ma_side": filter_ma_side,
    "ref_trend": filter_ref_trend,
    "squeeze": filter_squeeze,
    "session": filter_session,
    "atr_regime": filter_atr_regime,
    "volume": filter_volume,
    "rsi_bound": filter_rsi_bound,
    "ema_slope": filter_ema_slope,
}


def build_filter_masks(df: pd.DataFrame, filters: list[dict]):
    """Пересечение всех фильтров -> (mask_long, mask_short)."""
    n = len(df)
    ml = np.ones(n, dtype=bool)
    ms = np.ones(n, dtype=bool)
    for f in filters:
        fn = FILTER_BLOCKS.get(f["type"])
        if fn is None:
            raise ValueError(f"неизвестный фильтр: {f['type']}")
        a, b = fn(df, f.get("params", {}))
        ml &= a
        ms &= b
    return ml, ms
