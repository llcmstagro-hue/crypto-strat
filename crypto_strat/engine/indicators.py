"""
Индикаторы. ЖЕЛЕЗНОЕ ПРАВИЛО ФАЙЛА: всё causal.

Значение любого индикатора с индексом i рассчитано ТОЛЬКО по барам <= i.
Ни одна функция здесь не имеет права заглянуть в i+1. Это первый рубеж
защиты от lookahead (р.5.7); второй — в исполнителе (backtest.py), который
не даёт войти раньше следующего бара.

Каналы (Дончиан/Келтнер) отдаются в ДВУХ вариантах:
  *_incl — окно включает текущий бар (для оценки режима);
  *_prev — окно ЗАКАНЧИВАЕТСЯ на i-1 (для пробоя: уровень должен быть
           известен ДО того, как текущий бар его пробил).
Пробойные блоки обязаны брать *_prev, иначе получится тавтология
«бар пробил максимум, в который входит он сам».
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# БАЗА
# --------------------------------------------------------------------------- #
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_c = np.concatenate([[close[0]], close[:-1]])
    return np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])


def rma(x: np.ndarray, period: int) -> np.ndarray:
    """Сглаживание Уайлдера. Первые period-1 значений — NaN (данных не хватает)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < period or period < 1:
        return out
    out[period - 1] = x[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def ema(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < period or period < 1:
        return out
    out[period - 1] = x[:period].mean()
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def sma(x: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(x).rolling(period, min_periods=period).mean().to_numpy()


def atr(high, low, close, period: int) -> np.ndarray:
    return rma(true_range(high, low, close), period)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = rma(np.where(delta > 0, delta, 0.0), period)
    loss = rma(np.where(delta < 0, -delta, 0.0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(loss > 0, gain / loss, np.inf)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[np.isnan(gain) | np.isnan(loss)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# КАНАЛЫ
# --------------------------------------------------------------------------- #
def donchian(high, low, period: int, exclude_current: bool = True):
    """Канал Дончиана. exclude_current=True -> окно [i-period, i-1] (для пробоя)."""
    s_h, s_l = pd.Series(high), pd.Series(low)
    if exclude_current:
        s_h, s_l = s_h.shift(1), s_l.shift(1)
    up = s_h.rolling(period, min_periods=period).max().to_numpy()
    dn = s_l.rolling(period, min_periods=period).min().to_numpy()
    return up, dn


def keltner(high, low, close, ema_period: int, atr_period: int, mult: float,
            exclude_current: bool = True):
    """Канал Кельтнера вокруг EMA. exclude_current сдвигает границы на бар назад."""
    mid = ema(close, ema_period)
    a = atr(high, low, close, atr_period)
    up, dn = mid + mult * a, mid - mult * a
    if exclude_current:
        up = np.concatenate([[np.nan], up[:-1]])
        dn = np.concatenate([[np.nan], dn[:-1]])
        mid = np.concatenate([[np.nan], mid[:-1]])
    return up, mid, dn


# --------------------------------------------------------------------------- #
# СВИНГИ (фракталы)
# --------------------------------------------------------------------------- #
def find_swings(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Индексы свинг-хаёв/лоу. Пивот в точке i ПОДТВЕРЖДАЕТСЯ только на баре
    i+right — до этого момента система о нём не знает."""
    n = len(high)
    sh, sl = [], []
    for i in range(left, n - right):
        wh = high[i - left: i + right + 1]
        wl = low[i - left: i + right + 1]
        if high[i] == wh.max() and (wh == high[i]).sum() == 1:
            sh.append(i)
        if low[i] == wl.min() and (wl == low[i]).sum() == 1:
            sl.append(i)
    return sh, sl


def last_confirmed_swings(n: int, sh_idx, sl_idx, right: int):
    """Для каждого бара j — индекс последнего свинга, УЖЕ подтверждённого к бару j.
    Это и есть то, что стратегия имеет право знать в момент j."""
    last_sh = np.full(n, -1, dtype=int)
    last_sl = np.full(n, -1, dtype=int)
    p_h = p_l = 0
    cur_h = cur_l = -1
    for j in range(n):
        while p_h < len(sh_idx) and sh_idx[p_h] + right <= j:
            cur_h = sh_idx[p_h]; p_h += 1
        while p_l < len(sl_idx) and sl_idx[p_l] + right <= j:
            cur_l = sl_idx[p_l]; p_l += 1
        last_sh[j], last_sl[j] = cur_h, cur_l
    return last_sh, last_sl


# --------------------------------------------------------------------------- #
# СТАРШИЙ ТФ БЕЗ LOOKAHEAD
# --------------------------------------------------------------------------- #
def htf_resample_causal(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Агрегация в старший ТФ (factor баров в один) с ПРАВИЛЬНЫМ выравниванием.

    Значение старшего бара становится известно только в момент его ЗАКРЫТИЯ,
    поэтому результат сдвигается так, что на баре i лежит последний ПОЛНОСТЬЮ
    ЗАКРЫТЫЙ старший бар. Иначе фильтр тренда старшего ТФ подглядывает в будущее —
    классический и очень тихий баг.
    """
    n = len(df)
    grp = np.arange(n) // factor
    agg = pd.DataFrame({
        "open": df["open"].groupby(grp).transform("first"),
        "high": df["high"].groupby(grp).transform("max"),
        "low": df["low"].groupby(grp).transform("min"),
        "close": df["close"].groupby(grp).transform("last"),
    })
    # бар группы g закрыт на позиции (g+1)*factor - 1; до этого использовать нельзя
    closed = np.full(n, -1, dtype=int)
    for i in range(n):
        g = i // factor
        closed[i] = g - 1 if i < (g + 1) * factor - 1 else g
    out = pd.DataFrame(index=range(n), columns=["open", "high", "low", "close"], dtype=float)
    last_pos = np.where(closed >= 0, (closed + 1) * factor - 1, -1)
    valid = last_pos >= 0
    for c in ("open", "high", "low", "close"):
        vals = agg[c].to_numpy()
        col = np.full(n, np.nan)
        col[valid] = vals[last_pos[valid]]
        out[c] = col
    return out


def htf_trend_causal(df: pd.DataFrame, factor: int, ema_period: int) -> np.ndarray:
    """+1 / -1 / 0 — направление тренда старшего ТФ, известное на баре i.

    Считается по закрытиям старшего ТФ, сдвинутым так, что на i лежит последний
    ЗАКРЫТЫЙ старший бар (см. htf_resample_causal).
    """
    htf = htf_resample_causal(df, factor)
    close = htf["close"].to_numpy()
    # EMA по последовательности старших закрытий, разложенная обратно на младшие бары
    uniq_pos = np.arange(factor - 1, len(df), factor)
    if len(uniq_pos) < ema_period + 1:
        return np.zeros(len(df), dtype=int)
    htf_closes = df["close"].to_numpy()[uniq_pos]
    e = ema(htf_closes, ema_period)

    out = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        g = i // factor
        gi = g - 1 if i < (g + 1) * factor - 1 else g
        if gi < 0 or gi >= len(e) or np.isnan(e[gi]) or np.isnan(close[i]):
            continue
        out[i] = 1 if htf_closes[gi] > e[gi] else -1
    return out
