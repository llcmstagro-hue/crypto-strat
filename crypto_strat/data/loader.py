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


def normalize_ts_to_ms(ts: pd.Series) -> pd.Series:
    """Приводит таймстампы к миллисекундам, определяя единицы по величине.

    Разные источники отдают секунды, миллисекунды или микросекунды, и перепутать
    их дорого: ошибка в 1000 раз ломает выравнивание по границе ТФ и расчёт
    funding по 8-часовым границам — молча, без исключений.
    Ориентиры (для дат 2000-2100): с ~1e9, мс ~1e12, мкс ~1e15.
    """
    if ts.empty:
        return ts
    mx = float(ts.max())
    if mx < 1e11:            # секунды
        return ts * 1000
    if mx > 1e14:            # микросекунды
        return ts // 1000
    return ts                # уже миллисекунды


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
    df["ts"] = normalize_ts_to_ms(df["ts"]).astype("int64")
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


def resample_ohlcv(df: pd.DataFrame, tf_from: str, tf_to: str,
                   require_complete: bool = True) -> pd.DataFrame:
    """Агрегация свечей в старший ТФ.

    Два требования, без которых результат тихо испортится:
      * выравнивание по АБСОЛЮТНОЙ сетке времени (ts // tf_ms), а не по позиции
        в массиве: иначе после дырки в данных все старшие бары поедут, и,
        например, дневная свеча перестанет открываться в 00:00 UTC;
      * `require_complete` выбрасывает неполные группы. Старший бар, собранный
        из 2 часов вместо 4, — это не бар, а огрызок с заниженным диапазоном:
        он занижает ATR и делает стопы нереально узкими.
    """
    if tf_from not in TF_MS or tf_to not in TF_MS:
        raise DataError(f"неизвестный ТФ: {tf_from} -> {tf_to}")
    step = TF_MS[tf_to] // TF_MS[tf_from]
    if TF_MS[tf_to] % TF_MS[tf_from] or step < 2:
        raise DataError(f"{tf_from} не складывается в {tf_to} нацело")

    g = df["ts"].to_numpy() // TF_MS[tf_to]
    agg = df.groupby(g).agg(
        ts=("ts", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        _n=("close", "size"))
    agg["ts"] = agg.index.to_numpy() * TF_MS[tf_to]      # ровно граница ТФ
    if require_complete:
        agg = agg[agg["_n"] == step]
    out = agg.drop(columns="_n").reset_index(drop=True)
    out["dt_utc"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    out = out[["ts", "dt_utc", "open", "high", "low", "close", "volume"]]
    out.attrs["tf"] = tf_to
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
