"""
Импорт готовых CSV в формат конвейера (когда данные приходят не от загрузчика).

Делает четыре вещи, каждая из которых уже один раз спасала прогон:
  1. приводит таймстампы к миллисекундам (присланные файлы бывают в секундах —
     ошибка в 1000 раз молча ломает выравнивание по ТФ и начисление funding);
  2. переименовывает в формат `<PREFIX>_<SYMBOL>_<TF>.csv`, который ищет
     загрузчик корзины;
  3. прогоняет проверки качества р.5.6, включая детектор мёртвого фида
     (объём упал на порядки при внешне правдоподобных ценах);
  4. по флагу собирает старшие ТФ (H4/D) из H1 — с выравниванием по
     абсолютной сетке времени и отбрасыванием неполных баров.

Запуск:
    python import_csv.py /путь/к/файлам/*.csv
    python import_csv.py ./upload --resample 4h 1d
    python import_csv.py ./upload --truncate ETHUSDT=2026-03-01
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

from crypto_strat.data.loader import load_ohlcv_csv, resample_ohlcv, TF_MS
import download_data as dd


def parse_name(path: str) -> tuple[str, str] | None:
    """Достаёт (символ, ТФ) из имени файла. Префиксы-мусор игнорируются."""
    m = re.search(r"([A-Z]{2,10}USDT)[_-]?(1h|4h|1d|1H|4H|1D)?\.csv$",
                  os.path.basename(path), re.IGNORECASE)
    if not m:
        return None
    sym = m.group(1).upper()
    tf = (m.group(2) or "1h").lower()
    return sym, tf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="файлы или папки с CSV")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--prefix", default="USER")
    ap.add_argument("--resample", nargs="*", default=[], choices=["4h", "1d"],
                    help="собрать старшие ТФ из H1")
    ap.add_argument("--truncate", nargs="*", default=[],
                    help="обрезать инструмент по дате: SYMBOL=YYYY-MM-DD")
    args = ap.parse_args()

    cut = {}
    for item in args.truncate:
        sym, date = item.split("=")
        cut[sym.upper()] = int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p)
                     else sorted(glob.glob(p)))
    if not files:
        print("Не нашёл ни одного CSV.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    rows, imported = [], {}

    for path in files:
        parsed = parse_name(path)
        if not parsed:
            print(f"! пропускаю {os.path.basename(path)}: не разобрал символ/ТФ")
            continue
        sym, tf = parsed
        try:
            df = load_ohlcv_csv(path, tf=tf, strict=False)
        except Exception as e:
            print(f"! {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue

        if sym in cut:
            before = len(df)
            df = df[df["ts"] < cut[sym]].reset_index(drop=True)
            print(f"  {sym}: обрезан, {before} -> {len(df)} баров")

        out = os.path.join(args.out, f"{args.prefix}_{sym}_{tf}.csv")
        df.to_csv(out, index=False)
        rows.append((os.path.basename(out), dd.quality_report(df, tf)))
        if tf == "1h":
            imported[sym] = df
        print(f"  {sym} {tf}: {len(df)} баров, "
              f"{df['dt_utc'].iloc[0].date()} -> {df['dt_utc'].iloc[-1].date()}")

    for tf_to in args.resample:
        for sym, h1 in imported.items():
            agg = resample_ohlcv(h1, "1h", tf_to)
            out = os.path.join(args.out, f"{args.prefix}_{sym}_{tf_to}.csv")
            agg.to_csv(out, index=False)
            rows.append((os.path.basename(out), dd.quality_report(agg, tf_to)))
            print(f"  {sym}: H1 -> {tf_to}, {len(agg)} баров")

    dd.print_quality_table(rows)

    bad = [n for n, r in rows if dd.verdict(r) not in ("OK", "OK (единичные пропуски)")]
    if bad:
        print("\n⚠️  Файлы с замечаниями — разобраться ДО прогона фильтра:")
        for n in bad:
            print(f"   - {n}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
