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


ENTRY_BLOCKS = {
    "order_block": entry_order_block,
    "rsi_threshold": entry_rsi_threshold,
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


def filter_volume(df: pd.DataFrame, p: dict):
    """Объём выше своей средней — подтверждение интереса."""
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


FILTER_BLOCKS = {
    "htf_trend": filter_htf_trend,
    "ma_side": filter_ma_side,
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
