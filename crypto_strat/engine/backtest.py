"""
Модуль [2] — единый бэктест-движок.

Три вещи, ради которых он написан честно, и на которых обычно врут:

1. LOOKAHEAD. Сигнал возникает на ЗАКРЫТИИ бара i, исполнение — не раньше
   бара i+1. Это не соглашение, а свойство кода: исполнитель начинает
   рассматривать заявку с индекса signal_bar+1 и физически не имеет доступа
   к бару, на котором сигнал был сформирован. Индикаторы causal (indicators.py),
   фильтр старшего ТФ использует только ЗАКРЫТЫЕ старшие бары.

2. ИЗДЕРЖКИ. Комиссия + проскальзывание берутся с нотионала на КАЖДОЙ стороне,
   round-trip >= 0.3% зашит в конфиг. Плюс funding по факту пересечения
   8-часовых границ во время удержания. Ключевой эффект: издержки считаются
   в деньгах и лишь потом переводятся в R. Поэтому стратегия с микроскопическим
   стопом получает гигантский нотионал на ту же единицу риска — и издержки
   съедают её результат ровно так же, как в реальности. Это то, что автоматически
   убивает «красивые PF на мелком стопе» из р.4.

3. ВНУТРИБАРОВАЯ НЕОПРЕДЕЛЁННОСТЬ. Если бар задел и стоп, и тейк — считаем СТОП.
   Всегда. Пессимистично и не подглядывает в тиковые данные, которых нет.

Одна позиция на инструмент одновременно (без пирамид и без хеджа — прямое
следствие правил CFT). Сигналы, пришедшие в позицию, считаются и логируются,
но не исполняются.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .blocks import ENTRY_BLOCKS, OrderIntent, build_filter_masks
from .config import StrategyConfig

FUNDING_INTERVAL_MS = 8 * 3_600_000
TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


# --------------------------------------------------------------------------- #
# РЕЗУЛЬТАТ
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: np.ndarray                 # эквити по барам, включая нереализованное
    ts: np.ndarray
    symbol: str
    config: StrategyConfig
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def R(self) -> np.ndarray:
        return (self.trades["R"].to_numpy() if len(self.trades)
                else np.array([], dtype=float))


# --------------------------------------------------------------------------- #
# ВСПОМОГАТЕЛЬНОЕ
# --------------------------------------------------------------------------- #
def _resolve_stop(cfg: StrategyConfig, intent: OrderIntent, entry: float,
                  ctx: dict) -> float | None:
    """Цена защитного стопа. ВАЖНО: любые входные данные берутся на signal_bar,
    то есть известны ДО входа."""
    spec = cfg.stop
    stype = spec["type"]
    p = spec.get("params", {})
    d = intent.direction
    sb = intent.signal_bar

    if stype == "block":
        bs = intent.stop_spec
        if bs["type"] == "level":
            return float(bs["price"])
        if bs["type"] == "atr":
            return entry - d * float(bs["mult"]) * float(bs["atr"])
        return None

    if stype == "atr":
        a = ctx["atr"][sb]
        if np.isnan(a):
            return None
        return entry - d * float(p.get("mult", 2.0)) * float(a)

    if stype == "pct":
        return entry * (1.0 - d * float(p.get("pct", 0.01)))

    if stype == "structure":
        lb = int(p.get("lookback", 20))
        lo = max(0, sb - lb + 1)
        buf = float(p.get("buffer", 0.0005))
        if d > 0:
            return float(ctx["low"][lo:sb + 1].min()) * (1 - buf)
        return float(ctx["high"][lo:sb + 1].max()) * (1 + buf)

    if stype == "channel":
        period = int(p.get("period", 20))
        if period not in ctx["donch_cache"]:
            ctx["donch_cache"][period] = ind.donchian(
                ctx["high"], ctx["low"], period, exclude_current=True)
        up, dn = ctx["donch_cache"][period]
        lvl = dn[sb] if d > 0 else up[sb]
        return None if np.isnan(lvl) else float(lvl)

    raise ValueError(f"неизвестный тип стопа: {stype}")


def _funding_cost(funding: pd.DataFrame | None, default_rate: float,
                  t_in: int, t_out: int, notional: float, direction: int) -> float:
    """Funding за удержание (р.5.5). Начисляется на каждой пройденной 8-часовой
    границе. Лонг при положительной ставке ПЛАТИТ."""
    if t_out <= t_in:
        return 0.0
    first = ((t_in // FUNDING_INTERVAL_MS) + 1) * FUNDING_INTERVAL_MS
    if first > t_out:
        return 0.0
    boundaries = np.arange(first, t_out + 1, FUNDING_INTERVAL_MS, dtype="int64")
    if funding is None or funding.empty:
        rates = np.full(len(boundaries), default_rate)
    else:
        f_ts = funding["ts"].to_numpy()
        f_rt = funding["funding_rate"].to_numpy()
        pos = np.searchsorted(f_ts, boundaries, side="right") - 1
        rates = np.where(pos >= 0, f_rt[np.clip(pos, 0, len(f_rt) - 1)], default_rate)
    return float(notional * direction * rates.sum())


# --------------------------------------------------------------------------- #
# ДВИЖОК
# --------------------------------------------------------------------------- #
def run_backtest(df: pd.DataFrame,
                 cfg: StrategyConfig,
                 symbol: str = "?",
                 funding: pd.DataFrame | None = None,
                 initial_equity: float = 25_000.0) -> BacktestResult:
    n = len(df)
    if n < 50:
        return BacktestResult(_empty_trades(), np.array([initial_equity]),
                              np.array([0]), symbol, cfg,
                              {"reason": "слишком мало баров"})

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ts = df["ts"].to_numpy("int64")
    tf = df.attrs.get("tf") or cfg.timeframe
    tf_ms = TF_MS.get(tf, int(np.median(np.diff(ts))) if n > 2 else 3_600_000)

    # --- сигналы ---
    block = ENTRY_BLOCKS.get(cfg.entry["type"])
    if block is None:
        raise ValueError(f"неизвестный входной блок: {cfg.entry['type']}")
    intents: list[OrderIntent] = block(df, cfg.entry.get("params", {}))
    n_raw = len(intents)

    # --- фильтры и направление ---
    mask_long, mask_short = build_filter_masks(df, cfg.filters)
    allow_long = cfg.direction in ("long", "both")
    allow_short = cfg.direction in ("short", "both")

    kept = []
    for it in intents:
        if it.direction > 0 and not (allow_long and mask_long[it.signal_bar]):
            continue
        if it.direction < 0 and not (allow_short and mask_short[it.signal_bar]):
            continue
        kept.append(it)
    intents = sorted(kept, key=lambda x: x.signal_bar)
    n_after_filters = len(intents)

    # --- контекст для стопов/выходов ---
    stop_atr_p = int(cfg.stop.get("params", {}).get("period", 14))
    exit_p = cfg.exit.get("params", {})
    ctx = {
        "high": h, "low": l, "close": c,
        "atr": ind.atr(h, l, c, stop_atr_p),
        "donch_cache": {},
    }
    trail_atr = (ind.atr(h, l, c, int(exit_p.get("period", 14)))
                 if cfg.exit["type"] == "trailing_atr" else None)

    # --- Conqueror: трейлинг от экстремального ЗАКРЫТИЯ с кумулятивным сужением ---
    conq = None
    if cfg.exit["type"] == "conqueror_trail":
        from .blocks import conqueror_factors
        _, conq_changes, _ = conqueror_factors(df, exit_p)
        conq = {
            "atr": ind.atr(h, l, c, int(exit_p.get("atr_period", 40))),
            "changes": conq_changes,
            "base": float(exit_p.get("base_mult", 2.0)),
            "narrow": float(exit_p.get("narrow_factor", 2.0 / 3.0)),
            "min_mult": float(exit_p.get("min_mult", 0.1)),
        }
    struct_ema = (ind.ema(c, int(exit_p.get("ema_period", 50)))
                  if cfg.exit["type"] == "structure" else None)

    cost_rate = cfg.cost_rate_per_side()
    default_funding = float(cfg.costs["funding_default_8h"])
    risk_pct = float(cfg.sizing["risk_pct"])
    max_lev = float(cfg.sizing["max_leverage"])
    rr = float(exit_p.get("rr", 2.0))
    max_bars = int(exit_p.get("max_bars", 500))
    time_bars = int(exit_p.get("bars", max_bars))
    be_at_r = exit_p.get("breakeven_at_r")

    # --- заявки по бару активации ---
    by_activation: dict[int, list[OrderIntent]] = {}
    for it in intents:
        by_activation.setdefault(it.signal_bar + 1, []).append(it)

    equity = initial_equity
    equity_curve = np.full(n, initial_equity, dtype=float)
    active: list[OrderIntent] = []
    pos = None
    trades: list[dict] = []
    n_dropped_in_pos = 0
    n_no_stop = 0
    n_expired = 0
    n_inverted = 0

    for k in range(n):
        # 1) новые заявки становятся живыми ровно на баре signal_bar+1
        if k in by_activation:
            if pos is None:
                active.extend(by_activation[k])
            else:
                n_dropped_in_pos += len(by_activation[k])

        # 2) снятие просроченных и инвалидированных заявок
        if active:
            still = []
            for it in active:
                if k > it.signal_bar + it.valid_bars:
                    n_expired += 1
                    continue
                inv = it.invalidate
                if inv and k > it.signal_bar:
                    lvl, side = inv.get("level"), inv.get("side")
                    if side == "below" and c[k - 1] < lvl:
                        continue
                    if side == "above" and c[k - 1] > lvl:
                        continue
                still.append(it)
            active = still

        # 3) управление открытой позицией
        if pos is not None:
            closed = _manage(pos, k, o, h, l, c, ts, tf_ms, trail_atr, struct_ema,
                             exit_p, be_at_r, conq=conq)
            if closed is not None:
                tr = _close_trade(pos, closed, cost_rate, funding, default_funding,
                                  tf_ms, ts, equity)
                equity = tr["equity_after"]
                trades.append(tr)
                pos = None
                active = []                      # сигналы, накопившиеся в позиции, не берём
                equity_curve[k] = equity
                continue
            equity_curve[k] = equity + _unrealized(pos, c[k])

        # 4) попытка входа (только вне позиции)
        if pos is None:
            equity_curve[k] = equity
            fill = None
            for it in list(active):
                f = _try_fill(it, k, o, h, l)
                if f is not None:
                    fill = (it, f)
                    break
            if fill is not None:
                it, entry_price = fill
                stop_price = _resolve_stop(cfg, it, entry_price, ctx)
                # ИНВАРИАНТ: защитный стоп обязан лежать на ПРОИГРЫШНОЙ стороне
                # входа (ниже для лонга, выше для шорта).
                #
                # Структурные стопы (канал, свинг, край зоны) этого не гарантируют:
                # при входе против движения уровень запросто оказывается по другую
                # сторону цены. Без проверки abs() превращает такой стоп в
                # ПРИБЫЛЬНЫЙ выход — «стоп-лосс», приносящий +R. Winrate улетает
                # к 97%, PF к 20, и это выглядит как находка, а не как поломка.
                inverted = stop_price is not None and (
                    (it.direction > 0 and stop_price >= entry_price) or
                    (it.direction < 0 and stop_price <= entry_price))
                stop_dist = (abs(entry_price - stop_price)
                             if stop_price is not None else 0.0)
                if stop_price is None or stop_dist <= 0 or inverted:
                    if inverted:
                        n_inverted += 1
                    else:
                        n_no_stop += 1
                    active.remove(it)
                else:
                    risk_amount = equity * risk_pct
                    qty = risk_amount / stop_dist
                    max_qty = equity * max_lev / entry_price
                    capped = qty > max_qty
                    qty = min(qty, max_qty)
                    pos = {
                        "direction": it.direction, "entry_bar": k,
                        "entry_price": entry_price, "stop": stop_price,
                        "init_stop": stop_price, "stop_dist": stop_dist,
                        "qty": qty, "risk": qty * stop_dist,
                        "tp": (entry_price + it.direction * rr * stop_dist
                               if cfg.exit["type"] == "fixed_rr" else None),
                        "equity_before": equity, "capped": capped,
                        "max_bars": min(max_bars, time_bars),
                        "exit_type": cfg.exit["type"],
                        "mae": 0.0, "mfe": 0.0, "pending_exit": None,
                        "be_done": False, "block": it.meta.get("block", "?"),
                        # состояние Conqueror: экстремальное ЗАКРЫТИЕ и счётчик
                        # смен знака факторов с момента входа
                        "conq_extreme": float(c[k]), "conq_flips": 0,
                    }
                    active = []
                    # позиция может быть закрыта тем же баром — проверяем сразу
                    closed = _manage(pos, k, o, h, l, c, ts, tf_ms, trail_atr,
                                     struct_ema, exit_p, be_at_r, just_entered=True,
                                     conq=conq)
                    if closed is not None:
                        tr = _close_trade(pos, closed, cost_rate, funding,
                                          default_funding, tf_ms, ts, equity)
                        equity = tr["equity_after"]
                        trades.append(tr)
                        pos = None
                    equity_curve[k] = (equity if pos is None
                                       else equity + _unrealized(pos, c[k]))

    # позиция, оставшаяся открытой на конце данных, ЗАКРЫВАЕТСЯ по последнему
    # закрытию и помечается — иначе её убыток «исчезнет» из статистики
    if pos is not None:
        tr = _close_trade(pos, (n - 1, c[n - 1], "end_of_data"), cost_rate,
                          funding, default_funding, tf_ms, ts, equity)
        equity = tr["equity_after"]
        trades.append(tr)
        equity_curve[n - 1] = equity

    tdf = pd.DataFrame(trades) if trades else _empty_trades()
    if len(tdf):
        tdf["entry_ts"] = ts[tdf["entry_bar"].to_numpy()]
        tdf["exit_ts"] = ts[tdf["exit_bar"].to_numpy()]
        tdf["entry_dt"] = pd.to_datetime(tdf["entry_ts"], unit="ms", utc=True)
        tdf["exit_dt"] = pd.to_datetime(tdf["exit_ts"], unit="ms", utc=True)

    diagnostics = {
        "signals_raw": n_raw,
        "signals_after_filters": n_after_filters,
        "dropped_in_position": n_dropped_in_pos,
        "expired_orders": n_expired,
        "rejected_no_stop": n_no_stop,
        "rejected_inverted_stop": n_inverted,
        "leverage_capped": int(tdf["capped"].sum()) if len(tdf) else 0,
        "bars": n,
        "round_trip_bps": cfg.round_trip_bps(),
        "tf": tf,
    }
    return BacktestResult(tdf, equity_curve, ts, symbol, cfg, diagnostics)


# --------------------------------------------------------------------------- #
def _empty_trades() -> pd.DataFrame:
    cols = ["direction", "entry_bar", "exit_bar", "entry_price", "exit_price",
            "stop_price", "qty", "notional", "gross_pnl", "fees", "funding",
            "net_pnl", "R", "ret_pct", "bars_held", "exit_reason", "risk",
            "equity_before", "equity_after", "mae_R", "mfe_R", "capped", "block"]
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})


def _unrealized(pos: dict, price: float) -> float:
    return pos["qty"] * (price - pos["entry_price"]) * pos["direction"]


def _try_fill(it: OrderIntent, k: int, o, h, l) -> float | None:
    """Заполнение заявки на баре k.

    Лимит исполняется по своей цене ИЛИ ЛУЧШЕ: если бар открылся уже за
    уровнем (гэп сквозь заявку), сделка проходит по открытию. Брать в таком
    случае саму цену заявки нельзя — она может лежать выше high бара, то есть
    вне его диапазона: это не «пессимистично», это несуществующая цена.
    Стоп-заявка наоборот исполняется по своей цене ИЛИ ХУЖЕ — проскок на гэпе
    работает против нас.
    """
    if k < it.signal_bar + 1:
        return None
    d = it.direction
    if it.kind == "market":
        return float(o[k]) if k == it.signal_bar + 1 else None
    if it.kind == "limit":
        if d > 0 and l[k] <= it.price:
            return float(min(it.price, o[k]))
        if d < 0 and h[k] >= it.price:
            return float(max(it.price, o[k]))
        return None
    if it.kind == "stop":
        if d > 0 and h[k] >= it.price:
            return float(max(it.price, o[k]))
        if d < 0 and l[k] <= it.price:
            return float(min(it.price, o[k]))
        return None
    return None


def _manage(pos: dict, k: int, o, h, l, c, ts, tf_ms, trail_atr, struct_ema,
            exit_p: dict, be_at_r, just_entered: bool = False, conq: dict | None = None):
    """Ведение позиции на баре k. Возвращает (exit_bar, exit_price, reason) или None.

    Порядок ВНУТРИ бара зафиксирован и пессимистичен:
      отложенный выход по открытию -> стоп -> тейк.
    Если бар задел и стоп, и тейк — засчитывается СТОП.
    """
    d = pos["direction"]

    # отложенный выход (сигнал был на закрытии предыдущего бара) исполняется по открытию
    if pos["pending_exit"] is not None and not just_entered:
        reason = pos["pending_exit"]
        return (k, float(o[k]), reason)

    if not just_entered:
        # перевод в безубыток
        if be_at_r is not None and not pos["be_done"]:
            fav = (h[k] - pos["entry_price"]) * d if d > 0 else (pos["entry_price"] - l[k])
            if fav >= float(be_at_r) * pos["stop_dist"]:
                pos["stop"] = pos["entry_price"]
                pos["be_done"] = True

    # экскурсии в R
    fav = ((h[k] - pos["entry_price"]) if d > 0 else (pos["entry_price"] - l[k])) / pos["stop_dist"]
    adv = ((l[k] - pos["entry_price"]) if d > 0 else (pos["entry_price"] - h[k])) / pos["stop_dist"]
    pos["mfe"] = max(pos["mfe"], float(fav))
    pos["mae"] = min(pos["mae"], float(adv))

    # стоп. Различаем НАЧАЛЬНЫЙ защитный стоп и подтянутый трейлингом:
    # выход по трейлингу выше входа — это нормальная прибыль, а не «прибыльный
    # стоп-лосс». Одна метка на оба случая делает диагностику нечитаемой.
    if (d > 0 and l[k] <= pos["stop"]) or (d < 0 and h[k] >= pos["stop"]):
        moved = abs(pos["stop"] - pos["init_stop"]) > 1e-12
        return (k, float(pos["stop"]), "trail_stop" if moved else "stop")

    # тейк
    if pos["tp"] is not None:
        if (d > 0 and h[k] >= pos["tp"]) or (d < 0 and l[k] <= pos["tp"]):
            return (k, float(pos["tp"]), "take_profit")

    bars_held = k - pos["entry_bar"]

    # --- Conqueror: трейлинг от экстремального ЗАКРЫТИЯ + кумулятивное сужение ---
    # Коэффициент ATR уменьшается на треть при КАЖДОЙ смене знака любого из трёх
    # факторов — это не бинарный weak/strong, а пошаговое сужение, и оно
    # накапливается за время сделки: 2.00 -> 1.33 -> 0.89 -> 0.59 -> ...
    # Стоп двигается ТОЛЬКО в сторону прибыли, поэтому сужение может лишь
    # подтянуть его, но не отпустить обратно.
    if pos["exit_type"] == "conqueror_trail" and conq is not None:
        a = conq["atr"][k]
        if not np.isnan(a):
            if not just_entered:
                pos["conq_flips"] += int(conq["changes"][k])
            pos["conq_extreme"] = (max(pos["conq_extreme"], float(c[k])) if d > 0
                                   else min(pos["conq_extreme"], float(c[k])))
            mult = max(conq["base"] * (conq["narrow"] ** pos["conq_flips"]),
                       conq["min_mult"])
            cand = pos["conq_extreme"] - d * mult * a
            pos["stop"] = max(pos["stop"], cand) if d > 0 else min(pos["stop"], cand)
            pos["conq_mult"] = mult

    # трейлинг: новый стоп рассчитывается по ЗАКРЫТИЮ бара k и действует с k+1
    if pos["exit_type"] == "trailing_atr" and trail_atr is not None:
        a = trail_atr[k]
        if not np.isnan(a):
            mult = float(exit_p.get("mult", 2.5))
            cand = c[k] - d * mult * a
            pos["stop"] = max(pos["stop"], cand) if d > 0 else min(pos["stop"], cand)

    # выход по структуре: сигнал на закрытии -> исполнение по открытию следующего
    if pos["exit_type"] == "structure" and struct_ema is not None:
        e = struct_ema[k]
        if not np.isnan(e) and ((d > 0 and c[k] < e) or (d < 0 and c[k] > e)):
            pos["pending_exit"] = "structure"

    # выход по времени / предохранитель по числу баров
    if bars_held >= pos["max_bars"]:
        pos["pending_exit"] = "time"

    return None


def _close_trade(pos: dict, closed, cost_rate: float, funding, default_funding: float,
                 tf_ms: int, ts, equity: float) -> dict:
    exit_bar, exit_price, reason = closed
    d = pos["direction"]
    qty = pos["qty"]
    notional_in = qty * pos["entry_price"]
    notional_out = qty * exit_price

    gross = qty * (exit_price - pos["entry_price"]) * d
    fees = (notional_in + notional_out) * cost_rate

    t_in = int(ts[pos["entry_bar"]])
    t_out = int(ts[exit_bar]) + tf_ms
    fund = _funding_cost(funding, default_funding, t_in, t_out, notional_in, d)

    net = gross - fees - fund
    risk = pos["risk"]
    return {
        "direction": d,
        "entry_bar": pos["entry_bar"], "exit_bar": exit_bar,
        "entry_price": pos["entry_price"], "exit_price": exit_price,
        "stop_price": pos["init_stop"],
        "qty": qty, "notional": notional_in,
        "gross_pnl": gross, "fees": fees, "funding": fund, "net_pnl": net,
        "R": net / risk if risk > 0 else 0.0,
        "ret_pct": net / pos["equity_before"],
        "bars_held": exit_bar - pos["entry_bar"],
        "exit_reason": reason,
        "risk": risk,
        "equity_before": pos["equity_before"],
        "equity_after": pos["equity_before"] + net,
        "mae_R": pos["mae"], "mfe_R": pos["mfe"],
        "capped": bool(pos["capped"]),
        "block": pos["block"],
        "conq_flips": int(pos.get("conq_flips", 0)),
        "conq_mult": float(pos.get("conq_mult", np.nan)),
        "stop_dist_pct": pos["stop_dist"] / pos["entry_price"],
    }
