"""
Загрузчик рыночных данных под конвейер (модуль [0]).

Расширение download_h1_data.py:
  * таймфреймы H1 / H4 / D (было только H1);
  * bulk-режим data.binance.vision — помесячные дампы, самый чистый источник (р.5.2);
  * ccxt-режим — резерв и дозагрузка свежего хвоста, Bybit для форварда (р.5.3);
  * загрузка funding history для перпов (р.5.5);
  * проверки качества данных из р.5.6 с отчётом таблицей.

Роли источников (р.5.1):
  Binance bulk  -> бэктест и валидация (модули 0-3): глубина и чистота.
  Bybit  ccxt   -> форвард и исполнение (модуль 4): совпадение с площадкой.

Запуск:
    pip install pandas numpy requests ccxt

    # основной путь: bulk-дампы Binance USDT-перпов, все ТФ, с funding
    python download_data.py --source bulk --tf 1h 4h 1d --start 2022-01 --funding

    # данные площадки под форвард
    python download_data.py --source ccxt --exchange bybit --tf 1h 4h 1d

    # прогнать проверки качества по уже лежащим в ./data CSV
    python download_data.py --check-only

Результат: ./data/BINANCE_BTCUSDT_1h.csv  и  ./data/BINANCE_BTCUSDT_funding.csv
Формат CSV (его же читает движок):
    ts,dt_utc,open,high,low,close,volume
    ts   — миллисекунды UTC, время ОТКРЫТИЯ бара
    бар в файле присутствует только если он ЗАКРЫТ (р.5.7, защита от lookahead)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# КОНФИГ
# --------------------------------------------------------------------------- #
BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# ccxt пишет перпы иначе, чем bulk-дампы
CCXT_SYMBOL = {s: f"{s[:-4]}/USDT:USDT" for s in BASKET}

BULK_BASE = "https://data.binance.vision/data/futures/um"
FUNDING_INTERVAL_MS = 8 * 3_600_000  # перпы платят раз в 8 часов

OUT_DIR = "./data"

# Колонки klines в дампах Binance (12 полей, заголовка нет в старых файлах)
KLINE_COLS = [
    "ts", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def http_get(url: str, retries: int = 4, backoff: float = 2.0, timeout: int = 60):
    """GET с экспоненциальными повторами. Возвращает bytes или None на 404."""
    import requests

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"не смог скачать {url}: {type(e).__name__}: {e}") from e
            time.sleep(backoff * (2 ** (attempt - 1)))
    return None


def month_range(start: str, end: str | None = None):
    """Итератор 'YYYY-MM' от start до end включительно (end по умолчанию — прошлый месяц)."""
    y, m = (int(x) for x in start.split("-"))
    if end is None:
        now = datetime.now(timezone.utc)
        ey, em = (now.year, now.month)
    else:
        ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def day_range_of_current_month():
    """Дни текущего месяца до вчера включительно (дневные дампы появляются с лагом)."""
    now = datetime.now(timezone.utc)
    d = now.replace(day=1)
    out = []
    while d.date() < now.date():
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# BULK: data.binance.vision
# --------------------------------------------------------------------------- #
def _read_zip_csv(blob: bytes, cols: list[str]) -> pd.DataFrame:
    """Распаковывает единственный CSV из zip. Дампы бывают с заголовком и без."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    head = raw[:200].decode("utf-8", "ignore").lower()
    has_header = "open_time" in head or "open time" in head or "calc_time" in head
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else cols,
    )
    if has_header:
        df.columns = cols[: len(df.columns)]
    return df


def _verify_checksum(blob: bytes, url: str) -> bool:
    """Проверяет SHA256 против .CHECKSUM рядом с архивом. Нет файла — не блокируем."""
    try:
        chk = http_get(url + ".CHECKSUM", retries=2)
    except Exception:
        return True
    if not chk:
        return True
    expected = chk.decode().split()[0].strip()
    return hashlib.sha256(blob).hexdigest() == expected


def fetch_bulk_klines(symbol: str, tf: str, start: str, end: str | None) -> pd.DataFrame:
    """Собирает klines из помесячных дампов + дневные за текущий месяц."""
    frames: list[pd.DataFrame] = []
    missing_months: list[str] = []

    for ym in month_range(start, end):
        url = f"{BULK_BASE}/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{ym}.zip"
        blob = http_get(url)
        if blob is None:
            # нормально для месяцев ДО листинга перпа (SOL/DOGE — с 2021)
            missing_months.append(ym)
            continue
        if not _verify_checksum(blob, url):
            print(f"    ! контрольная сумма не сошлась: {ym}, пропускаю файл")
            continue
        frames.append(_read_zip_csv(blob, KLINE_COLS))

    if end is None:
        for day in day_range_of_current_month():
            url = f"{BULK_BASE}/daily/klines/{symbol}/{tf}/{symbol}-{tf}-{day}.zip"
            try:
                blob = http_get(url, retries=2)
            except Exception:
                blob = None
            if blob is None:
                continue
            frames.append(_read_zip_csv(blob, KLINE_COLS))

    if missing_months:
        first, last = missing_months[0], missing_months[-1]
        print(f"    (нет дампов за {len(missing_months)} мес.: {first}..{last} — "
              f"вероятно, перп ещё не был запущен)")
    if not frames:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    return normalize_klines(df)


def fetch_bulk_funding(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Собирает историю funding rate из помесячных дампов (р.5.5)."""
    cols = ["calc_time", "funding_interval_hours", "funding_rate"]
    frames = []
    for ym in month_range(start, end):
        url = f"{BULK_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        try:
            blob = http_get(url, retries=2)
        except Exception:
            blob = None
        if blob is None:
            continue
        frames.append(_read_zip_csv(blob, cols))
    if not frames:
        return pd.DataFrame(columns=["ts", "dt_utc", "funding_rate"])

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"calc_time": "ts"})
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna(subset=["ts", "funding_rate"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt_utc"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    return df[["ts", "dt_utc", "funding_rate"]]


# --------------------------------------------------------------------------- #
# CCXT (резерв + данные площадки под форвард)
# --------------------------------------------------------------------------- #
def fetch_ccxt_klines(exchange_id: str, symbol: str, tf: str, since_ms: int) -> pd.DataFrame:
    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True,
                                     "options": {"defaultType": "swap"}})
    ex.load_markets()
    sym = CCXT_SYMBOL[symbol] if CCXT_SYMBOL[symbol] in ex.markets else symbol
    tf_ms = TF_MS[tf]
    now = ex.milliseconds()
    since, rows, last = since_ms, [], None

    while since < now:
        batch = None
        for attempt in range(1, 6):
            try:
                batch = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
                break
            except Exception as e:
                print(f"    [retry {attempt}/5] {type(e).__name__}: {e}")
                time.sleep(2.0 * attempt)
        if not batch:
            break
        if last is not None and batch[-1][0] <= last:
            break
        rows += batch
        last = batch[-1][0]
        since = last + tf_ms
        time.sleep(ex.rateLimit / 1000.0)

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return normalize_klines(df)


def fetch_ccxt_funding(exchange_id: str, symbol: str, since_ms: int) -> pd.DataFrame:
    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True,
                                     "options": {"defaultType": "swap"}})
    ex.load_markets()
    sym = CCXT_SYMBOL[symbol] if CCXT_SYMBOL[symbol] in ex.markets else symbol
    if not ex.has.get("fetchFundingRateHistory"):
        print(f"    ! {exchange_id} не отдаёт fetchFundingRateHistory")
        return pd.DataFrame(columns=["ts", "dt_utc", "funding_rate"])

    out, since, now = [], since_ms, ex.milliseconds()
    while since < now:
        try:
            batch = ex.fetch_funding_rate_history(sym, since=since, limit=200)
        except Exception as e:
            print(f"    ! funding: {type(e).__name__}: {e}")
            break
        if not batch:
            break
        out += [(r["timestamp"], r["fundingRate"]) for r in batch]
        nxt = batch[-1]["timestamp"] + FUNDING_INTERVAL_MS
        if nxt <= since:
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000.0)

    if not out:
        return pd.DataFrame(columns=["ts", "dt_utc", "funding_rate"])
    df = pd.DataFrame(out, columns=["ts", "funding_rate"])
    df = df.dropna().drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt_utc"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    return df[["ts", "dt_utc", "funding_rate"]]


# --------------------------------------------------------------------------- #
# НОРМАЛИЗАЦИЯ
# --------------------------------------------------------------------------- #
def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Единый формат: числовые OHLCV, ts в мс UTC, дедуп, сортировка."""
    keep = ["ts", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=keep)

    # часть дампов Binance с 2025 отдаёт микросекунды — приводим к мс
    if len(df) and df["ts"].max() > 1e14:
        df["ts"] = df["ts"] // 1000

    df["ts"] = df["ts"].astype("int64")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts", "dt_utc", "open", "high", "low", "close", "volume"]]


def drop_unclosed_bar(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Убирает формирующуюся свечу — прямой источник lookahead (р.5.7)."""
    if df.empty:
        return df
    tf_ms = TF_MS[tf]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return df[df["ts"] + tf_ms <= now_ms].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# ПРОВЕРКИ КАЧЕСТВА (р.5.6)
# --------------------------------------------------------------------------- #
def quality_report(df: pd.DataFrame, tf: str) -> dict:
    """Все проверки р.5.6 разом. Возвращает словарь метрик качества."""
    tf_ms = TF_MS[tf]
    rep = {
        "bars": len(df), "gaps": 0, "missing_bars": 0, "dups": 0,
        "ohlc_broken": 0, "zero_vol": 0, "nonmonotonic": 0,
        "misaligned": 0, "first": None, "last": None, "coverage_pct": 0.0,
    }
    if df.empty:
        return rep

    ts = df["ts"].to_numpy()
    rep["dups"] = int(len(ts) - len(np.unique(ts)))
    rep["nonmonotonic"] = int((np.diff(ts) <= 0).sum())

    diffs = np.diff(ts)
    gap_mask = diffs > tf_ms
    rep["gaps"] = int(gap_mask.sum())
    rep["missing_bars"] = int(np.round(diffs[gap_mask] / tf_ms - 1).sum()) if gap_mask.any() else 0

    # выравнивание по границе ТФ: дневная свеча обязана открываться в 00:00 UTC
    rep["misaligned"] = int((ts % tf_ms != 0).sum())

    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    rep["ohlc_broken"] = int((
        (h < l) | (h < o) | (h < c) | (l > o) | (l > c) | (o <= 0) | (c <= 0)
    ).sum())
    rep["zero_vol"] = int((df["volume"].to_numpy() <= 0).sum())

    rep["first"] = df["dt_utc"].iloc[0]
    rep["last"] = df["dt_utc"].iloc[-1]
    expected = (ts[-1] - ts[0]) / tf_ms + 1
    rep["coverage_pct"] = round(100.0 * len(ts) / expected, 3) if expected > 0 else 0.0
    return rep


def verdict(rep: dict) -> str:
    """Светофор по качеству. Массовые пропуски = перекачать."""
    if rep["bars"] == 0:
        return "ПУСТО"
    bad = (rep["dups"] or rep["nonmonotonic"] or rep["ohlc_broken"] or rep["misaligned"])
    if bad:
        return "БРАК"
    if rep["coverage_pct"] < 99.0:
        return "ДЫРЯВО"
    if rep["missing_bars"] > 0:
        return "OK (единичные пропуски)"
    return "OK"


def print_quality_table(rows: list[tuple]) -> None:
    print("\n" + "=" * 108)
    print("КАЧЕСТВО ДАННЫХ (р.5.6)")
    print("=" * 108)
    print(f"{'файл':<28}{'баров':>8}{'период':>26}{'покр.%':>9}"
          f"{'проп':>7}{'дубл':>6}{'OHLC':>6}{'вырав':>7}  вердикт")
    print("-" * 108)
    for name, rep in rows:
        span = (f"{rep['first'].date()}..{rep['last'].date()}"
                if rep["first"] is not None else "-")
        print(f"{name:<28}{rep['bars']:>8}{span:>26}{rep['coverage_pct']:>9.2f}"
              f"{rep['missing_bars']:>7}{rep['dups']:>6}{rep['ohlc_broken']:>6}"
              f"{rep['misaligned']:>7}  {verdict(rep)}")
    print("=" * 108)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def run_check_only(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        print(f"Папки {os.path.abspath(out_dir)} нет — сначала скачай данные:\n"
              f"  python download_data.py --source bulk --tf 1h 4h 1d "
              f"--start 2022-01 --funding")
        return
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".csv") or "funding" in fn:
            continue
        tf = fn.rsplit("_", 1)[-1].replace(".csv", "")
        if tf not in TF_MS:
            continue
        df = pd.read_csv(os.path.join(out_dir, fn))
        df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        rows.append((fn, quality_report(df, tf)))
    if not rows:
        print(f"В {os.path.abspath(out_dir)} нет CSV со свечами.")
        return
    print_quality_table(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Загрузчик данных под конвейер стратегий")
    p.add_argument("--source", choices=["bulk", "ccxt"], default="bulk",
                   help="bulk = data.binance.vision (для бэктеста), ccxt = live API (для форварда)")
    p.add_argument("--exchange", default="bybit", help="биржа для режима ccxt")
    p.add_argument("--tf", nargs="+", default=["1h", "4h", "1d"], choices=list(TF_MS))
    p.add_argument("--symbols", nargs="+", default=BASKET)
    p.add_argument("--start", default="2022-01", help="начало истории, YYYY-MM")
    p.add_argument("--end", default=None, help="конец истории, YYYY-MM (по умолчанию — сейчас)")
    p.add_argument("--funding", action="store_true", help="дополнительно скачать funding history")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--check-only", action="store_true",
                   help="ничего не качать, только прогнать проверки качества по ./data")
    args = p.parse_args()

    if args.check_only:
        run_check_only(args.out)
        return 0

    os.makedirs(args.out, exist_ok=True)
    tag = "BINANCE" if args.source == "bulk" else args.exchange.upper()
    since_ms = int(datetime.strptime(args.start, "%Y-%m").replace(tzinfo=timezone.utc)
                   .timestamp() * 1000)

    print(f"Источник: {args.source} | ТФ: {' '.join(args.tf)} | с {args.start}")
    print(f"Инструменты: {' '.join(args.symbols)}")
    print("-" * 70)

    quality_rows, failures = [], []
    for sym in args.symbols:
        for tf in args.tf:
            print(f"\n>>> {sym} {tf}")
            try:
                if args.source == "bulk":
                    df = fetch_bulk_klines(sym, tf, args.start, args.end)
                else:
                    df = fetch_ccxt_klines(args.exchange, sym, tf, since_ms)
            except Exception as e:
                print(f"    ! ОШИБКА: {type(e).__name__}: {e}")
                failures.append(f"{sym} {tf}: {e}")
                continue

            df = drop_unclosed_bar(df, tf)
            if df.empty:
                print("    ! пусто")
                failures.append(f"{sym} {tf}: пусто")
                continue

            path = os.path.join(args.out, f"{tag}_{sym}_{tf}.csv")
            save(df, path)
            rep = quality_report(df, tf)
            quality_rows.append((os.path.basename(path), rep))
            print(f"    баров: {len(df)} | {rep['first']} -> {rep['last']}")
            print(f"    сохранено: {path}")

        if args.funding:
            print(f"\n>>> {sym} funding")
            try:
                fdf = (fetch_bulk_funding(sym, args.start, args.end) if args.source == "bulk"
                       else fetch_ccxt_funding(args.exchange, sym, since_ms))
                if fdf.empty:
                    print("    ! funding пуст")
                    failures.append(f"{sym} funding: пусто")
                else:
                    fpath = os.path.join(args.out, f"{tag}_{sym}_funding.csv")
                    save(fdf, fpath)
                    ann = fdf["funding_rate"].mean() * 3 * 365 * 100
                    print(f"    записей: {len(fdf)} | средний funding "
                          f"{fdf['funding_rate'].mean()*100:.4f}% / 8ч (~{ann:.1f}% годовых)")
                    print(f"    сохранено: {fpath}")
            except Exception as e:
                print(f"    ! ОШИБКА funding: {type(e).__name__}: {e}")
                failures.append(f"{sym} funding: {e}")

    if quality_rows:
        print_quality_table(quality_rows)
    if failures:
        print("\nНЕ ЗАГРУЖЕНО:")
        for f in failures:
            print(f"  - {f}")
    print(f"\nГотово. Данные в: {os.path.abspath(args.out)}")
    return 1 if failures and not quality_rows else 0


if __name__ == "__main__":
    sys.exit(main())
