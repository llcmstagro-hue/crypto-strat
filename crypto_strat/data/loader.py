"""
Загрузка данных в движок (модуль [0] -> [2]).

Читает ровно тот CSV, который пишет download_data.py:
    ts,dt_utc,open,high,low,close,volume

Плюс опциональный funding-файл:
    ts,dt_utc,funding_rate

Если реальных CSV ещё нет — движок работает на synthetic.py, но любой
результат оттуда помечается как НЕ ЗНАЧИМЫЙ (см. PROGRESS.md, Этап 4).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
REQUIRED = ["ts", "open", "high", "low", "close", "volume"]

BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]


class DataError(RuntimeError):
    pass


def load_ohlcv_csv(path: str, tf: str | None = None, strict: bool = True) -> pd.DataFrame:
    """Читает CSV со свечами и валидирует его. strict=True -> брак роняет загрузку.

    Валидация тут не косметика: битые бары (high < low, дубли, немонотонность)
    тихо ломают исполнение стопов и дают фантомные метрики.
    """
    if not os.path.exists(path):
        raise DataError(f"нет файла: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"{path}: нет колонок {missing}")

    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df["ts"] = df["ts"].astype("int64")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    broken = int(((h < l) | (h < o) | (h < c) | (l > o) | (l > c) | (c <= 0)).sum())
    if broken:
        msg = f"{path}: битых баров по OHLC: {broken}"
        if strict:
            raise DataError(msg)
        print("  ! " + msg)

    tf = tf or infer_tf(path, df)
    if tf and tf in TF_MS:
        misaligned = int((df["ts"].to_numpy() % TF_MS[tf] != 0).sum())
        if misaligned and strict:
            raise DataError(f"{path}: {misaligned} баров не выровнены по границе {tf}")

    df.attrs["tf"] = tf
    df.attrs["source"] = os.path.basename(path)
    return df[["ts", "dt_utc", "open", "high", "low", "close", "volume"]]


def infer_tf(path: str, df: pd.DataFrame | None = None) -> str | None:
    """ТФ из имени файла (…_1h.csv), иначе — из медианного шага таймстампов."""
    m = re.search(r"_(1h|4h|1d)\.csv$", os.path.basename(path))
    if m:
        return m.group(1)
    if df is not None and len(df) > 10:
        step = int(np.median(np.diff(df["ts"].to_numpy())))
        for k, v in TF_MS.items():
            if v == step:
                return k
    return None


def load_funding_csv(path: str) -> pd.DataFrame | None:
    """История funding. Нет файла — вернём None, движок возьмёт плоскую ставку."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "funding_rate" not in df.columns or "ts" not in df.columns:
        return None
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna(subset=["ts", "funding_rate"])
    df["ts"] = df["ts"].astype("int64")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)[["ts", "funding_rate"]]


def load_basket(data_dir: str = "./data", tf: str = "1h",
                symbols: list[str] | None = None,
                prefix: str | None = None,
                with_funding: bool = True) -> dict[str, dict]:
    """Грузит корзину: {symbol: {'ohlcv': df, 'funding': df|None}}.

    prefix — 'BINANCE' / 'BYBIT'; None = взять первый подходящий файл.
    Отсутствующие инструменты молча пропускаются: кросс-инструментальный
    барьер сам разберётся, хватает ли покрытия корзины.
    """
    symbols = symbols or BASKET
    out: dict[str, dict] = {}
    if not os.path.isdir(data_dir):
        return out

    files = os.listdir(data_dir)
    for sym in symbols:
        cands = [f for f in files
                 if f.endswith(f"_{sym}_{tf}.csv")
                 and (prefix is None or f.startswith(prefix + "_"))]
        if not cands:
            continue
        path = os.path.join(data_dir, sorted(cands)[0])
        try:
            ohlcv = load_ohlcv_csv(path, tf=tf, strict=False)
        except DataError as e:
            print(f"  ! {sym}: {e}")
            continue

        funding = None
        if with_funding:
            for f in files:
                if f.endswith(f"_{sym}_funding.csv") and (prefix is None or f.startswith(prefix + "_")):
                    funding = load_funding_csv(os.path.join(data_dir, f))
                    break
        out[sym] = {"ohlcv": ohlcv, "funding": funding}
    return out


def slice_by_fraction(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Срез по доле длины (для train/test-сплитов). Всегда по ВРЕМЕНИ, не случайно —
    перемешивать временной ряд нельзя, это утечка будущего в прошлое."""
    n = len(df)
    a, b = int(n * start), int(n * end)
    return df.iloc[a:b].reset_index(drop=True)


def slice_by_ts(df: pd.DataFrame, ts_from: int, ts_to: int) -> pd.DataFrame:
    m = (df["ts"] >= ts_from) & (df["ts"] < ts_to)
    return df[m].reset_index(drop=True)
