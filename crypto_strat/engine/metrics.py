"""
Метрики. Считаем то, что имеет смысл, и отдельно — то, что важно под CFT.

Осознанное разделение:
  * метрики в R    — свойство САМОГО СЕТАПА, не зависят от сайзинга;
  * метрики в %    — свойство КОНФИГУРАЦИИ РИСКА, именно по ним челлендж
                     проваливают (р.1: проваливают размером позиции, не сетапом).

Приёмочные цифры трейдера (3%/мес, просадка 4%) считаются здесь, но НИГДЕ
не используются как критерий отбора или цель оптимизации — только как
финальный фильтр после форварда (р.2, р.7).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MS_YEAR = 365.25 * 24 * 3_600_000


def _safe(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else default


def trade_metrics(R: np.ndarray) -> dict:
    """Метрики по распределению R. Основа всего отбора."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    if n == 0:
        return {"n_trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "expectancy_R": 0.0, "std_R": 0.0, "sharpe_trade": 0.0,
                "total_R": 0.0, "max_dd_R": 0.0, "skew": 0.0, "kurtosis": 3.0,
                "max_loss_streak": 0, "avg_win_R": 0.0, "avg_loss_R": 0.0}

    wins, losses = R[R > 0], R[R < 0]
    gross_w, gross_l = wins.sum(), -losses.sum()
    eq = np.cumsum(R)
    dd = np.maximum.accumulate(eq) - eq
    sd = R.std(ddof=1) if n > 1 else 0.0

    streak = best = 0
    for r in R:
        streak = streak + 1 if r < 0 else 0
        best = max(best, streak)

    mean = R.mean()
    m3 = ((R - mean) ** 3).mean()
    m4 = ((R - mean) ** 4).mean()
    sd_pop = R.std(ddof=0)

    return {
        "n_trades": n,
        "winrate": float(len(wins) / n),
        # PF при нулевых убытках не «бесконечен», а неопределён — отдаём 0,
        # чтобы такая выборка не выигрывала сортировку по PF
        "profit_factor": _safe(gross_w / gross_l) if gross_l > 0 else 0.0,
        "expectancy_R": float(mean),
        "std_R": float(sd),
        "sharpe_trade": _safe(mean / sd) if sd > 0 else 0.0,
        "total_R": float(eq[-1]),
        "max_dd_R": float(dd.max()),
        "skew": _safe(m3 / sd_pop ** 3) if sd_pop > 0 else 0.0,
        "kurtosis": _safe(m4 / sd_pop ** 4, 3.0) if sd_pop > 0 else 3.0,
        "max_loss_streak": int(best),
        "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
    }


def equity_metrics(equity: np.ndarray, ts: np.ndarray,
                   initial: float | None = None) -> dict:
    """Метрики эквити в процентах. Просадка считается по кривой, ВКЛЮЧАЮЩЕЙ
    нереализованное — иначе внутрисделочная просадка исчезает из отчёта,
    а дневной лимит CFT её как раз видит."""
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "monthly_return_pct": 0.0,
                "years": 0.0, "cft_static_breach": False, "worst_day_pct": 0.0,
                "cft_daily_breach": False}

    init = initial if initial is not None else equity[0]
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, np.nan)
    max_dd = float(np.nanmax(dd))

    years = max((ts[-1] - ts[0]) / MS_YEAR, 1e-9)
    total_ret = equity[-1] / init - 1.0
    # среднемесячная геометрическая
    months = years * 12
    monthly = ((equity[-1] / init) ** (1 / months) - 1.0) if equity[-1] > 0 and months > 0 else -1.0

    # дневная динамика — прямая проверка лимитов CFT (р.1)
    day = pd.to_datetime(ts, unit="ms", utc=True).floor("D")
    s = pd.Series(equity, index=day)
    day_open = s.groupby(level=0).first()
    day_min = s.groupby(level=0).min()
    day_loss = (day_min / day_open - 1.0)
    worst_day = float(day_loss.min()) if len(day_loss) else 0.0

    return {
        "return_pct": float(total_ret),
        "max_dd_pct": max_dd,
        "monthly_return_pct": float(monthly),
        "years": float(years),
        "worst_day_pct": worst_day,
        # статичная просадка CFT: пол эквити = 90% от НАЧАЛЬНОГО баланса
        "cft_static_breach": bool(equity.min() < init * 0.90),
        "cft_daily_breach": bool(worst_day <= -0.05),
    }


def full_metrics(result) -> dict:
    """Полный отчёт по одному прогону."""
    m = trade_metrics(result.R)
    m.update(equity_metrics(result.equity, result.ts))
    tdf = result.trades
    if len(tdf):
        years = max(m["years"], 1e-9)
        m["trades_per_year"] = float(len(tdf) / years)
        m["avg_bars_held"] = float(tdf["bars_held"].mean())
        m["median_stop_pct"] = float(tdf["stop_dist_pct"].median())
        m["total_fees"] = float(tdf["fees"].sum())
        m["total_funding"] = float(tdf["funding"].sum())
        m["gross_pnl"] = float(tdf["gross_pnl"].sum())
        m["net_pnl"] = float(tdf["net_pnl"].sum())
        gross = m["gross_pnl"]
        m["cost_share"] = float((m["total_fees"] + m["total_funding"]) / abs(gross)) if gross else 0.0
        m["long_share"] = float((tdf["direction"] > 0).mean())
    else:
        m.update({"trades_per_year": 0.0, "avg_bars_held": 0.0, "median_stop_pct": 0.0,
                  "total_fees": 0.0, "total_funding": 0.0, "gross_pnl": 0.0,
                  "net_pnl": 0.0, "cost_share": 0.0, "long_share": 0.0})
    m["symbol"] = result.symbol
    m["diagnostics"] = result.diagnostics
    return m


def pooled_metrics(results: list) -> dict:
    """Сводка по корзине: сделки складываются в общий пул.

    Пул честнее среднего по инструментам: инструмент с 5 сделками не должен
    весить столько же, сколько инструмент с 300.
    """
    if not results:
        return trade_metrics(np.array([]))
    R = np.concatenate([r.R for r in results]) if any(r.n_trades for r in results) else np.array([])
    m = trade_metrics(R)
    m["per_symbol"] = {r.symbol: trade_metrics(r.R) for r in results}
    m["symbols_with_trades"] = int(sum(1 for r in results if r.n_trades > 0))
    return m


def format_report(m: dict, label: str = "") -> str:
    """Человекочитаемый отчёт. Ничего не приукрашиваем."""
    L = []
    L.append(f"--- {label} ---")
    L.append(f"  сделок:        {m['n_trades']}")
    if m["n_trades"] == 0:
        return "\n".join(L) + "\n  (пусто)"
    L.append(f"  winrate:       {m['winrate']*100:.1f}%")
    L.append(f"  profit factor: {m['profit_factor']:.2f}")
    L.append(f"  expectancy:    {m['expectancy_R']:+.3f} R")
    L.append(f"  Sharpe/сделку: {m['sharpe_trade']:+.3f}")
    L.append(f"  итог:          {m['total_R']:+.1f} R")
    L.append(f"  макс. DD:      {m['max_dd_R']:.1f} R")
    if "max_dd_pct" in m:
        L.append(f"  макс. DD %:    {m['max_dd_pct']*100:.2f}%")
        L.append(f"  доходность:    {m['monthly_return_pct']*100:+.2f}% / мес "
                 f"(за {m['years']:.2f} г.)")
    if "cost_share" in m:
        L.append(f"  издержки:      {m['cost_share']*100:.1f}% от валовой прибыли")
    if "median_stop_pct" in m and m["median_stop_pct"]:
        L.append(f"  медиана стопа: {m['median_stop_pct']*100:.2f}% от цены")
    L.append(f"  макс. серия убытков: {m['max_loss_streak']}")
    return "\n".join(L)
