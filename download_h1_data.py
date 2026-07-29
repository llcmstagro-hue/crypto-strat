"""
Загрузчик H1-данных под агентную систему стратегий.

Качает часовые (1h) свечи по корзине из 5 перпов USDT за 3-4 года,
проверяет на пропуски, сохраняет в CSV с UTC-таймстампами.

Корзина: BTC, ETH, SOL, BNB, DOGE.

Биржа по умолчанию — Bybit (твоя торговая площадка под CFT).
Если история Bybit по какому-то инструменту окажется короче нужной,
переключись на Binance (EXCHANGE = "binance") — там глубже история.

Запуск (локально или в Colab):
    pip install ccxt pandas numpy
    python download_h1_data.py

Результат: папка ./data/ с файлами вида BYBIT_BTCUSDT_1h.csv
Эти CSV потом загружаешь в чат — на них строится движок и весь конвейер.
"""

from __future__ import annotations
import os
import time
import sys
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CONFIG = {
    "exchange": "bybit",          # "bybit" (торговая площадка) или "binance" (глубже история)
    "timeframe": "1h",
    "years": 4,                   # глубина истории в годах (3-4)
    "symbols": [                  # перпы USDT
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "BNB/USDT:USDT",
        "DOGE/USDT:USDT",
    ],
    "out_dir": "./data",
    "batch_limit": 1000,          # свечей за один запрос (лимит большинства бирж)
    "max_retries": 5,             # повторов при сетевой ошибке
    "retry_sleep": 3.0,           # пауза между повторами, сек
}

# Для Binance символы перпов записываются иначе — маппинг применяется автоматически.
BINANCE_SYMBOL_MAP = {
    "BTC/USDT:USDT": "BTC/USDT",
    "ETH/USDT:USDT": "ETH/USDT",
    "SOL/USDT:USDT": "SOL/USDT",
    "BNB/USDT:USDT": "BNB/USDT",
    "DOGE/USDT:USDT": "DOGE/USDT",
}


# --------------------------------------------------------------------------- #
# ЗАГРУЗКА
# --------------------------------------------------------------------------- #
def make_exchange(exchange_id: str):
    import ccxt
    klass = getattr(ccxt, exchange_id)
    # defaultType=future -> публичные данные по перпам, ключи не нужны
    opts = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    if exchange_id == "binance":
        opts["options"]["defaultType"] = "future"
    return klass(opts)


def resolve_symbol(exchange_id: str, symbol: str) -> str:
    if exchange_id == "binance":
        return BINANCE_SYMBOL_MAP.get(symbol, symbol)
    return symbol


def fetch_ohlcv_paginated(ex, symbol: str, timeframe: str, since_ms: int,
                          batch_limit: int, max_retries: int, retry_sleep: float) -> list[list]:
    """Тянет свечи постранично от since_ms до 'сейчас'. Без заглядывания в будущее:
    просто исторические бары, отсортированные по времени."""
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = since_ms
    all_rows: list[list] = []
    last_ts = None

    while since < now:
        rows = None
        for attempt in range(1, max_retries + 1):
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
                break
            except Exception as e:
                print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
                time.sleep(retry_sleep)
        if rows is None:
            print("    ! не удалось получить батч после всех повторов, прерываю инструмент.")
            break
        if not rows:
            break

        # защита от зацикливания, если биржа отдаёт тот же кусок
        if last_ts is not None and rows[-1][0] <= last_ts:
            break

        all_rows += rows
        last_ts = rows[-1][0]
        since = rows[-1][0] + tf_ms

        # мягкий троттлинг (в дополнение к enableRateLimit)
        time.sleep(ex.rateLimit / 1000.0)

    return all_rows


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


# --------------------------------------------------------------------------- #
# ПРОВЕРКА КАЧЕСТВА
# --------------------------------------------------------------------------- #
def check_gaps(df: pd.DataFrame, timeframe_ms: int) -> dict:
    """Ищет пропущенные бары. Для крипты (24/7) пропусков быть почти не должно;
    единичные — норма (обрыв связи биржи), массовые — повод перекачать."""
    ts = df["ts"].to_numpy()
    if len(ts) < 2:
        return {"gaps": 0, "missing_bars": 0, "dup": 0}
    diffs = np.diff(ts)
    expected = timeframe_ms
    gap_mask = diffs > expected
    missing = int(((diffs[gap_mask] / expected) - 1).round().sum()) if gap_mask.any() else 0
    return {
        "gaps": int(gap_mask.sum()),
        "missing_bars": missing,
        "dup": int((diffs == 0).sum()),
    }


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    cfg = CONFIG
    ex_id = cfg["exchange"]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    try:
        ex = make_exchange(ex_id)
        ex.load_markets()
    except Exception as e:
        print(f"Не удалось инициализировать биржу '{ex_id}': {e}")
        print("Проверь: pip install ccxt, доступ в интернет, имя биржи.")
        sys.exit(1)

    tf = cfg["timeframe"]
    tf_ms = ex.parse_timeframe(tf) * 1000
    since_ms = ex.milliseconds() - cfg["years"] * 365 * 24 * 60 * 60 * 1000

    print(f"Биржа: {ex_id} | ТФ: {tf} | глубина: {cfg['years']} лет")
    print(f"Старт истории: {pd.to_datetime(since_ms, unit='ms', utc=True)}")
    print("-" * 70)

    summary = []
    for symbol in cfg["symbols"]:
        sym = resolve_symbol(ex_id, symbol)
        print(f"\n>>> {sym}")
        if sym not in ex.markets:
            print(f"    ! {sym} нет на {ex_id}. Пропускаю. "
                  f"(Попробуй другую биржу или проверь символ.)")
            summary.append((sym, 0, None, None, "НЕТ НА БИРЖЕ"))
            continue

        rows = fetch_ohlcv_paginated(
            ex, sym, tf, since_ms,
            cfg["batch_limit"], cfg["max_retries"], cfg["retry_sleep"],
        )
        if not rows:
            print("    ! пусто.")
            summary.append((sym, 0, None, None, "ПУСТО"))
            continue

        df = to_frame(rows)
        q = check_gaps(df, tf_ms)

        # безопасное имя файла
        safe = sym.replace("/", "").replace(":", "")
        fname = f"{ex_id.upper()}_{safe}_{tf}.csv"
        fpath = os.path.join(cfg["out_dir"], fname)
        df.to_csv(fpath, index=False)

        first_dt = df["dt_utc"].iloc[0]
        last_dt = df["dt_utc"].iloc[-1]
        print(f"    свечей: {len(df)} | {first_dt} -> {last_dt}")
        print(f"    пропуски: {q['gaps']} (баров ~{q['missing_bars']}) | дубли: {q['dup']}")
        print(f"    сохранено: {fpath}")
        summary.append((sym, len(df), first_dt, last_dt, "OK"))

    # --- итоговая таблица ---
    print("\n" + "=" * 70)
    print("ИТОГ ЗАГРУЗКИ")
    print("=" * 70)
    for sym, n, a, b, status in summary:
        span = ""
        if a is not None and b is not None:
            span = f"{a.date()} -> {b.date()}"
        print(f"  {sym:<18} {n:>7} свечей  {span:<26} [{status}]")
    print("\nГотово. CSV лежат в:", os.path.abspath(cfg["out_dir"]))
    print("Загрузи эти файлы в чат — на них строим движок.")


if __name__ == "__main__":
    main()
