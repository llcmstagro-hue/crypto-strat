"""
Генератор синтетических рядов — ТОЛЬКО для проверки инструмента, не стратегий.

Зачем он вообще есть, если р.7 запрещает синтетику вместо реальных данных:
CLAUDE.md запрещает ОЦЕНИВАТЬ СТРАТЕГИИ на синтетике. Здесь другая задача —
откалибровать сам ФИЛЬТР (Этап 4). Для этого нужен ряд с ЗАРАНЕЕ ИЗВЕСТНЫМ
ответом, чего реальный рынок дать не может: на BTC никто не знает, есть ли
там эдж на самом деле, поэтому «фильтр пропустил» невозможно отличить от
«фильтр ошибся».

Два режима — это две половины теста инструмента:

  regime="edge"  — в цену ЗАШИТ настоящий persistent momentum (скрытая
                   марковская цепь режимов с дрейфом). Пробойная стратегия
                   обязана иметь тут положительную ожидаемость.
                   Проверяет ЧУВСТВИТЕЛЬНОСТЬ: честный эдж должен пройти.

  regime="noise" — чистое GBM с той же волатильностью и БЕЗ структуры.
                   Эджа нет ни у кого. Проверяет СПЕЦИФИЧНОСТЬ: если что-то
                   прошло фильтр здесь — в фильтре дыра (утечка/lookahead).

Бары строятся из под-шагов внутри свечи, поэтому OHLC согласован, а стопы
и тейки выбиваются реалистично, а не по нарисованным экстремумам.

⚠️ Метрики стратегий, полученные на этих данных, В ОТЧЁТАХ НЕ ИСПОЛЬЗУЮТСЯ
как доказательство эджа. Только реальные CSV из download_data.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
SUBSTEPS = 12  # под-шагов внутри одного бара


def make_ohlcv(n: int = 12_000,
               regime: str = "edge",
               seed: int = 7,
               tf: str = "1h",
               start_ts: int = 1_640_995_200_000,   # 2022-01-01 UTC
               price0: float = 30_000.0,
               vol_daily: float = 0.035,
               trend_strength: float = 0.10) -> pd.DataFrame:
    """Синтетические свечи.

    trend_strength — дрейф в трендовых состояниях, в долях волатильности бара.

    Значение по умолчанию 0.10 подобрано НАМЕРЕННО СЛАБЫМ: пробойная стратегия
    получает на таком ряду expectancy ~ +0.11R при PF ~ 1.17 и winrate ~34% —
    это масштаб РЕАЛЬНОГО эджа, а не рисованного. Калибровать фильтр на
    гигантском эдже (PF 15) бессмысленно: пропустить такой сумеет и сломанный
    фильтр, а нам нужно, чтобы он видел скромный честный сигнал.
    """
    if regime not in ("edge", "noise"):
        raise ValueError("regime: 'edge' или 'noise'")
    rng = np.random.default_rng(seed)

    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}[tf]
    sigma_bar = vol_daily / np.sqrt(bars_per_day)
    sigma_step = sigma_bar / np.sqrt(SUBSTEPS)

    # --- кластеризация волатильности (общая для обоих режимов) ---
    log_v = np.zeros(n)
    for i in range(1, n):
        log_v[i] = 0.97 * log_v[i - 1] + rng.normal(0, 0.12)
    vol_mult = np.exp(log_v - log_v.var() / 2)

    # --- дрейф ---
    if regime == "noise":
        drift = np.zeros(n)                      # никакой структуры
    else:
        # 3 состояния: тренд вверх / флэт / тренд вниз, высокая инерция
        P = np.array([[0.985, 0.015, 0.000],
                      [0.010, 0.980, 0.010],
                      [0.000, 0.015, 0.985]])
        mu = np.array([+trend_strength, 0.0, -trend_strength]) * sigma_bar
        state = 1
        drift = np.zeros(n)
        for i in range(n):
            state = rng.choice(3, p=P[state])
            drift[i] = mu[state]

    # --- под-шаги -> согласованный OHLC ---
    steps = rng.normal(0.0, 1.0, size=(n, SUBSTEPS))
    steps = steps * (sigma_step * vol_mult[:, None]) + (drift[:, None] / SUBSTEPS)

    log_path = np.log(price0) + np.cumsum(steps.reshape(-1))
    path = np.exp(log_path).reshape(n, SUBSTEPS)

    open_ = np.empty(n)
    open_[0] = price0
    open_[1:] = path[:-1, -1]
    close = path[:, -1]
    high = np.maximum(path.max(axis=1), np.maximum(open_, close))
    low = np.minimum(path.min(axis=1), np.minimum(open_, close))

    volume = rng.lognormal(3.0, 0.6, n) * vol_mult

    ts = start_ts + np.arange(n, dtype="int64") * TF_MS[tf]
    df = pd.DataFrame({
        "ts": ts,
        "dt_utc": pd.to_datetime(ts, unit="ms", utc=True),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })
    df.attrs["tf"] = tf
    df.attrs["synthetic"] = regime
    return df


def make_funding(df: pd.DataFrame, seed: int = 11, mean_rate: float = 0.0001) -> pd.DataFrame:
    """Синтетический funding: в среднем 0.01% / 8ч (лонги платят), с шумом.

    Это близко к историческому среднему по мейджорам и достаточно, чтобы
    издержка удержания не была нулевой в тестах движка.
    """
    rng = np.random.default_rng(seed)
    t0, t1 = int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])
    step = 8 * 3_600_000
    ts = np.arange(t0 - t0 % step, t1 + step, step, dtype="int64")
    rate = rng.normal(mean_rate, 0.00015, len(ts))
    return pd.DataFrame({"ts": ts, "funding_rate": rate})


def make_basket(symbols: list[str] | None = None, regime: str = "edge",
                n: int = 12_000, tf: str = "1h", seed: int = 7) -> dict[str, dict]:
    """Корзина синтетических инструментов — для проверки кросс-инструментального
    барьера. Разные seed = независимые ряды с одинаковой ПРИРОДОЙ (тот же режим),
    что имитирует «один и тот же эдж на разных инструментах»."""
    symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
    prices = {"BTCUSDT": 30_000.0, "ETHUSDT": 2_000.0, "SOLUSDT": 40.0,
              "BNBUSDT": 300.0, "DOGEUSDT": 0.12}
    vols = {"BTCUSDT": 0.032, "ETHUSDT": 0.038, "SOLUSDT": 0.055,
            "BNBUSDT": 0.035, "DOGEUSDT": 0.070}
    out = {}
    for k, sym in enumerate(symbols):
        d = make_ohlcv(n=n, regime=regime, seed=seed + 101 * k, tf=tf,
                       price0=prices.get(sym, 100.0),
                       vol_daily=vols.get(sym, 0.04))
        out[sym] = {"ohlcv": d, "funding": make_funding(d, seed=seed + 7 * k)}
    return out
