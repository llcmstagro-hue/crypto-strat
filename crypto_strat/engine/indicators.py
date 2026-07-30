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
# УРОВНИ ПРЕДЫДУЩЕГО ДНЯ
# --------------------------------------------------------------------------- #
DAY_MS = 86_400_000


def prev_day_levels(ts: np.ndarray, high: np.ndarray, low: np.ndarray):
    """Хай/лоу ПОСЛЕДНЕГО ПОЛНОСТЬЮ ЗАКРЫТОГО календарного дня (UTC).

    Возвращает (pdh, pdl) по барам: на баре i лежат экстремумы последнего дня,
    строго ПРЕДШЕСТВУЮЩЕГО дню самого бара i. На D1 это просто предыдущий бар,
    на H4 — агрегат шести баров вчерашнего дня.

    Причинность. День D-1 закрыт до того, как открылся первый бар дня D,
    поэтому уровень известен всем барам дня D целиком — включая самый первый.
    Никакого сдвига «на бар» тут не нужно и он был бы неверен: он лишил бы
    первый бар дня уровня, который на бирже к тому моменту уже существует.

    Дыры в данных обрабатываются честно: берётся последний ИМЕЮЩИЙСЯ день до
    текущего, а не «день минус один». Если вчерашних баров нет, уровень —
    от позавчера; выдумывать отсутствующий день нельзя, а обнулять уровень
    значило бы молча выкинуть сигналы после каждой дыры.
    """
    n = len(ts)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    if n == 0:
        return pdh, pdl
    day = (ts // DAY_MS).astype("int64")

    cur_day = day[0]
    cur_h, cur_l = -np.inf, np.inf         # копится текущий день
    prev_h, prev_l = np.nan, np.nan        # последний ЗАКРЫТЫЙ день
    for i in range(n):
        if day[i] != cur_day:
            # текущий день закончился -> он и становится «предыдущим»
            if np.isfinite(cur_h):
                prev_h, prev_l = cur_h, cur_l
            cur_day = day[i]
            cur_h, cur_l = -np.inf, np.inf
        pdh[i], pdl[i] = prev_h, prev_l    # значение ДО учёта текущего бара
        cur_h = max(cur_h, high[i])
        cur_l = min(cur_l, low[i])
    return pdh, pdl


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
    n = len(df)
    # закрытия ПОЛНЫХ старших баров лежат на позициях factor-1, 2*factor-1, ...
    uniq_pos = np.arange(factor - 1, n, factor)
    if len(uniq_pos) < ema_period + 1:
        return np.zeros(n, dtype=int)
    htf_closes = df["close"].to_numpy()[uniq_pos]
    e = ema(htf_closes, ema_period)

    i = np.arange(n)
    g = i // factor
    # старший бар g доступен только начиная с его последнего младшего бара;
    # до этого момента актуален предыдущий, уже закрытый
    gi = np.where((i + 1) % factor == 0, g, g - 1)
    ok = (gi >= 0) & (gi < len(e))
    gi_safe = np.clip(gi, 0, max(len(e) - 1, 0))
    ok &= ~np.isnan(e[gi_safe])

    out = np.zeros(n, dtype=int)
    out[ok] = np.where(htf_closes[gi_safe][ok] > e[gi_safe][ok], 1, -1)
    return out


# --------------------------------------------------------------------------- #
# ADX и СТОХАСТИК (нужны методам Куртни Смита)
# --------------------------------------------------------------------------- #
def adx(high, low, close, period: int = 14):
    """ADX по Уайлдеру. Возвращает (adx, plus_di, minus_di).

    Causal: значение на баре i использует только бары <= i. Смит применяет ADX
    двумя способами — как фильтр («берём сигнал, только если ADX выше вчерашнего»)
    и как выход Bishop («ADX поднялся выше 40 и начал опускаться»).
    """
    n = len(close)
    up = np.zeros(n)
    dn = np.zeros(n)
    up[1:] = high[1:] - high[:-1]
    dn[1:] = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = true_range(high, low, close)
    atr_ = rma(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * rma(plus_dm, period) / atr_
        mdi = 100.0 * rma(minus_dm, period) / atr_
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    dx[~np.isfinite(dx)] = np.nan
    return rma(np.nan_to_num(dx, nan=0.0), period), pdi, mdi


def stochastic(high, low, close, k_period: int = 14, k_smooth: int = 3,
               d_period: int = 3):
    """Стохастик %K/%D. Окно ВКЛЮЧАЕТ текущий бар — так он и определён
    (положение закрытия внутри диапазона последних k баров), это не lookahead."""
    s_h = pd.Series(high).rolling(k_period, min_periods=k_period).max().to_numpy()
    s_l = pd.Series(low).rolling(k_period, min_periods=k_period).min().to_numpy()
    rng = s_h - s_l
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * (close - s_l) / rng
    raw[~np.isfinite(raw)] = np.nan
    k = pd.Series(raw).rolling(k_smooth, min_periods=k_smooth).mean().to_numpy()
    d = pd.Series(k).rolling(d_period, min_periods=d_period).mean().to_numpy()
    return k, d
