"""
SMC Order Block backtester (crypto, BTC/ETH).

Что делает скрипт:
  1. Грузит OHLCV (через ccxt, публичные данные, ключи не нужны).
  2. Ищет свинги (фракталы) без заглядывания в будущее.
  3. Детектит BOS (break of structure) и связанный с ним Order Block.
  4. Ждёт возврата цены в зону OB -> вход, стоп за зоной, тейк по R/R.
  5. Считает метрики: winrate, profit factor, expectancy (R), макс. просадку.

Правила формализованы механически, чтобы тест был воспроизводимым.
Меняй параметры в CONFIG и сравнивай результат на in-sample / out-of-sample.

Запуск:
    pip install ccxt pandas numpy
    python smc_order_block_backtest.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — крути эти параметры и смотри, устойчива ли стратегия
# --------------------------------------------------------------------------- #
CONFIG = {
    "symbol": "BTC/USDT",     # или "ETH/USDT"
    "timeframe": "1h",        # "15m", "1h", "4h", "1d"
    "candles": 3000,          # сколько свечей истории тянуть
    "exchange": "binance",    # любая биржа, поддерживаемая ccxt

    "swing_left": 3,          # свечей слева для подтверждения пивота
    "swing_right": 3,         # свечей справа (задержка подтверждения)

    "ob_use_body": False,     # True -> зона OB = тело свечи, False -> весь диапазон
    "ob_max_age": 60,         # макс. срок жизни OB в свечах, потом снимаем
    "stop_buffer": 0.0005,    # буфер за границей OB (0.05%)
    "risk_reward": 2.0,       # тейк = вход +/- rr * риск
    "both_directions": True,  # тестить и лонги, и шорты

    "oos_split": 0.7,         # доля данных под in-sample (остальное — out-of-sample)
}


# --------------------------------------------------------------------------- #
# ЗАГРУЗКА ДАННЫХ
# --------------------------------------------------------------------------- #
def fetch_ohlcv(symbol: str, timeframe: str, limit: int, exchange_id: str) -> pd.DataFrame:
    """Тянет OHLCV через ccxt постранично."""
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = now - limit * tf_ms
    rows: list[list] = []
    while since < now and len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.tail(limit).reset_index(drop=True)


def make_synthetic(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Синтетические свечи для проверки логики без сети."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.006, n) + np.sin(np.arange(n) / 40) * 0.0015
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=n, freq="h"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(1, 100, n),
    })


# --------------------------------------------------------------------------- #
# СВИНГИ (фракталы)
# --------------------------------------------------------------------------- #
def find_swings(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Возвращает индексы подтверждённых свинг-хаёв и свинг-лоу.
    Пивот в точке i подтверждается только на свече i+right (без lookahead)."""
    n = len(high)
    swing_high, swing_low = [], []
    for i in range(left, n - right):
        win_h = high[i - left: i + right + 1]
        win_l = low[i - left: i + right + 1]
        if high[i] == win_h.max() and (win_h == high[i]).sum() == 1:
            swing_high.append(i)
        if low[i] == win_l.min() and (win_l == low[i]).sum() == 1:
            swing_low.append(i)
    return swing_high, swing_low


# --------------------------------------------------------------------------- #
# ДЕТЕКЦИЯ ORDER BLOCKS ЧЕРЕЗ BOS
# --------------------------------------------------------------------------- #
def detect_order_blocks(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Находит OB. Каждый OB 'создаётся' на свече BOS (confirm_idx),
    значит доступен для входа только со следующей свечи — без lookahead."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(df)
    left, right = cfg["swing_left"], cfg["swing_right"]
    sh_idx, sl_idx = find_swings(h, l, left, right)

    # для каждой свечи — последний свинг-уровень, ПОДТВЕРЖДЁННЫЙ к этому моменту
    last_sh = np.full(n, -1)   # индекс свинг-хая
    last_sl = np.full(n, -1)
    sh_ptr = sl_ptr = 0
    cur_sh = cur_sl = -1
    for j in range(n):
        # свинг в точке k подтверждён на k+right
        while sh_ptr < len(sh_idx) and sh_idx[sh_ptr] + right <= j:
            cur_sh = sh_idx[sh_ptr]; sh_ptr += 1
        while sl_ptr < len(sl_idx) and sl_idx[sl_ptr] + right <= j:
            cur_sl = sl_idx[sl_ptr]; sl_ptr += 1
        last_sh[j] = cur_sh
        last_sl[j] = cur_sl

    obs: list[dict] = []
    used_bull, used_bear = set(), set()

    for j in range(1, n):
        # --- бычий BOS: закрытие выше подтверждённого свинг-хая ---
        sh = last_sh[j]
        if sh != -1 and c[j] > h[sh] and sh not in used_bull:
            # ищем OB = последняя медвежья свеча между свинг-хаем и BOS
            ob = None
            for k in range(j, sh, -1):
                if c[k] < o[k]:            # медвежья свеча
                    ob = k; break
            if ob is not None:
                top = max(o[ob], c[ob]) if cfg["ob_use_body"] else h[ob]
                bottom = min(o[ob], c[ob]) if cfg["ob_use_body"] else l[ob]
                obs.append({"dir": "long", "created": j,
                            "top": top, "bottom": bottom})
                used_bull.add(sh)

        # --- медвежий BOS: закрытие ниже подтверждённого свинг-лоу ---
        sl = last_sl[j]
        if cfg["both_directions"] and sl != -1 and c[j] < l[sl] and sl not in used_bear:
            ob = None
            for k in range(j, sl, -1):
                if c[k] > o[k]:            # бычья свеча
                    ob = k; break
            if ob is not None:
                top = max(o[ob], c[ob]) if cfg["ob_use_body"] else h[ob]
                bottom = min(o[ob], c[ob]) if cfg["ob_use_body"] else l[ob]
                obs.append({"dir": "short", "created": j,
                            "top": top, "bottom": bottom})
                used_bear.add(sl)

    return obs


# --------------------------------------------------------------------------- #
# БЭКТЕСТ
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame, obs: list[dict], cfg: dict) -> pd.DataFrame:
    """Вход при возврате в зону OB. Результат каждой сделки — в R-мультипликаторах."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(df)
    buf = cfg["stop_buffer"]
    rr = cfg["risk_reward"]
    max_age = cfg["ob_max_age"]
    trades = []

    for ob in obs:
        start = ob["created"] + 1          # вход не раньше следующей свечи
        end = min(n, ob["created"] + max_age)
        top, bottom = ob["top"], ob["bottom"]
        entered = False

        for k in range(start, end):
            if ob["dir"] == "long":
                # инвалидция: пробили зону насквозь вниз до входа
                if c[k] < bottom:
                    break
                # цена коснулась зоны сверху -> лимитный вход по top
                if not entered and l[k] <= top:
                    entry = top
                    stop = bottom * (1 - buf)
                    risk = entry - stop
                    if risk <= 0:
                        break
                    tp = entry + rr * risk
                    entered = True
                    entry_idx = k
                    continue
                if entered:
                    hit_stop = l[k] <= stop
                    hit_tp = h[k] >= tp
                    if hit_stop and hit_tp:          # оба на одной свече -> считаем стоп
                        trades.append((ob["dir"], entry_idx, k, -1.0)); break
                    if hit_stop:
                        trades.append((ob["dir"], entry_idx, k, -1.0)); break
                    if hit_tp:
                        trades.append((ob["dir"], entry_idx, k, rr)); break
            else:  # short
                if c[k] > top:
                    break
                if not entered and h[k] >= bottom:
                    entry = bottom
                    stop = top * (1 + buf)
                    risk = stop - entry
                    if risk <= 0:
                        break
                    tp = entry - rr * risk
                    entered = True
                    entry_idx = k
                    continue
                if entered:
                    hit_stop = h[k] >= stop
                    hit_tp = l[k] <= tp
                    if hit_stop and hit_tp:
                        trades.append((ob["dir"], entry_idx, k, -1.0)); break
                    if hit_stop:
                        trades.append((ob["dir"], entry_idx, k, -1.0)); break
                    if hit_tp:
                        trades.append((ob["dir"], entry_idx, k, rr)); break

    return pd.DataFrame(trades, columns=["dir", "entry_idx", "exit_idx", "R"])


# --------------------------------------------------------------------------- #
# МЕТРИКИ
# --------------------------------------------------------------------------- #
def metrics(trades: pd.DataFrame, label: str = "") -> None:
    if trades.empty:
        print(f"[{label}] сделок нет — ослабь фильтры или дай больше данных.")
        return
    R = trades["R"].to_numpy()
    wins = R[R > 0]; losses = R[R < 0]
    equity = np.cumsum(R)
    peak = np.maximum.accumulate(equity)
    max_dd = (peak - equity).max()
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n===== РЕЗУЛЬТАТ [{label}] =====")
    print(f"Сделок:            {len(R)}")
    print(f"Winrate:           {len(wins)/len(R)*100:5.1f}%")
    print(f"Profit factor:     {pf:5.2f}")
    print(f"Expectancy:        {R.mean():+5.2f} R / сделку")
    print(f"Итог:              {equity[-1]:+6.1f} R")
    print(f"Макс. просадка:    {max_dd:5.1f} R")
    print(f"Лонг / Шорт:       {(trades['dir']=='long').sum()} / {(trades['dir']=='short').sum()}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, cfg: dict) -> None:
    split = int(len(df) * cfg["oos_split"])
    df_is = df.iloc[:split].reset_index(drop=True)
    df_oos = df.iloc[split:].reset_index(drop=True)

    for label, d in [("IN-SAMPLE", df_is), ("OUT-OF-SAMPLE", df_oos), ("ВСЁ", df)]:
        obs = detect_order_blocks(d, cfg)
        tr = backtest(d, obs, cfg)
        print(f"\n[{label}] найдено OB: {len(obs)}")
        metrics(tr, label)


if __name__ == "__main__":
    try:
        data = fetch_ohlcv(CONFIG["symbol"], CONFIG["timeframe"],
                           CONFIG["candles"], CONFIG["exchange"])
        print(f"Загружено {len(data)} свечей {CONFIG['symbol']} {CONFIG['timeframe']}")
    except Exception as e:
        print(f"ccxt недоступен ({e}). Тест на синтетике.")
        data = make_synthetic(CONFIG["candles"])

    run(data, CONFIG)
