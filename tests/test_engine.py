"""
Тесты движка. Главный из них — прямая проверка на lookahead.

Идея теста на заглядывание в будущее: обрезаем ряд на баре K и прогоняем
бэктест на обрезке. Если движок нигде не подглядывает вперёд, набор сделок,
ЗАКРЫВШИХСЯ до K, обязан совпасть посимвольно с тем, что даёт полный ряд.
Любая утечка будущего — незакрытая свеча, нецентрированный индикатор,
неправильно выровненный старший ТФ — немедленно разъедется в ценах входа
или в моменте выхода.

Запуск:
    python -m pytest tests/ -q      (если есть pytest)
    python tests/test_engine.py     (без pytest)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from crypto_strat.data.synthetic import make_ohlcv, make_funding
from crypto_strat.engine.config import StrategyConfig, ConfigError
from crypto_strat.engine.backtest import run_backtest, _funding_cost
from crypto_strat.engine import indicators as ind

CONFIGS = [
    {"name": "donchian", "entry": {"type": "donchian_breakout", "params": {"period": 20}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 200}},
     "filters": [{"type": "htf_trend", "params": {"factor": 4, "ema_period": 50}}]},
    {"name": "ob", "entry": {"type": "order_block", "params": {}},
     "stop": {"type": "block", "params": {}},
     "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 100}},
     "filters": []},
    {"name": "rsi2", "entry": {"type": "rsi_threshold", "params": {"period": 2, "oversold": 10.0,
                                                                   "overbought": 90.0}},
     "stop": {"type": "structure", "params": {"lookback": 20}},
     "exit": {"type": "structure", "params": {"ema_period": 10, "max_bars": 72}},
     "filters": [{"type": "ma_side", "params": {"period": 100}}]},
    {"name": "fvg", "entry": {"type": "fvg", "params": {"min_size_pct": 0.001, "max_age": 40}},
     "stop": {"type": "block", "params": {}},
     "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 80}},
     "filters": [{"type": "atr_regime", "params": {"min_pct": 0.001, "max_pct": 0.05}}]},
    # новые классы: волатильностный и межрыночный
    {"name": "squeeze", "entry": {"type": "squeeze_breakout",
                                  "params": {"squeeze_lookback": 120, "squeeze_pct": 0.3}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 200}},
     "filters": []},
    {"name": "ttm", "entry": {"type": "ttm_squeeze", "params": {}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 100}},
     "filters": []},
    {"name": "nr", "entry": {"type": "nr_expansion", "params": {"lookback": 7}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 200}},
     "filters": []},
    {"name": "consensus2", "entry": {"type": "consensus", "params": {
        "components": [{"type": "donchian_breakout", "params": {}},
                       {"type": "squeeze_breakout", "params": {}}],
        "window": 6, "min_agree": 2}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 200}},
     "filters": []},
    {"name": "consensus2of3", "entry": {"type": "consensus", "params": {
        "components": [{"type": "donchian_breakout", "params": {}},
                       {"type": "keltner_breakout", "params": {}},
                       {"type": "nr_expansion", "params": {}}],
        "window": 6, "min_agree": 2}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 100}},
     "filters": [{"type": "squeeze", "params": {"lookback": 120, "pct": 0.4}}]},
    {"name": "xmkt", "entry": {"type": "ref_momentum",
                               "params": {"ref": "REF", "lookback": 6, "threshold": 0.03}},
     "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
     "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0, "max_bars": 200}},
     "filters": [{"type": "ref_trend", "params": {"ref": "REF", "ema_period": 50}}]},
]


def _with_ref(df):
    """Прикрепляет ведущий инструмент — коррелированный, но не идентичный ряд."""
    ref = make_ohlcv(n=len(df), regime="edge", seed=999, price0=1000.0)
    ref = ref.iloc[:len(df)].copy()
    ref["ts"] = df["ts"].to_numpy()
    df = df.copy()
    df.attrs["refs"] = {"REF": ref[["ts", "close"]]}
    return df


def test_no_lookahead():
    """Обрезка ряда не должна менять уже состоявшиеся сделки."""
    base = make_ohlcv(n=8000, regime="edge", seed=17)
    df = _with_ref(base)
    fund = make_funding(df)
    cut = 5000

    for d in CONFIGS:
        cfg = StrategyConfig.from_dict(dict(d))
        full = run_backtest(df, cfg, "T", funding=fund)
        # обрезанный ряд получает ПОЛНЫЙ ведущий инструмент — ровно так и
        # бывает в бою (BTC длиннее альта). Если блок выравнивает не по
        # таймстампу, а по позиции, тест это поймает
        cut_df = df.iloc[:cut].reset_index(drop=True)
        cut_df.attrs["refs"] = df.attrs["refs"]
        trunc = run_backtest(cut_df, cfg, "T", funding=fund)

        # сравниваем только то, что закрылось с запасом до точки обрезки:
        # у обрезанного ряда последняя сделка принудительно закрыта по end_of_data
        margin = 300
        a = full.trades[full.trades["exit_bar"] < cut - margin] if full.n_trades else full.trades
        b = trunc.trades[trunc.trades["exit_bar"] < cut - margin] if trunc.n_trades else trunc.trades

        assert len(a) == len(b), (
            f"{cfg.name}: разное число сделок до обрезки — {len(a)} vs {len(b)}. "
            f"Это lookahead: полный ряд «знает» то, чего не знает обрезанный.")
        if len(a) == 0:
            continue
        for col in ("entry_bar", "exit_bar", "direction"):
            assert (a[col].to_numpy() == b[col].to_numpy()).all(), \
                f"{cfg.name}: расходится колонка {col} — lookahead"
        for col in ("entry_price", "exit_price", "R"):
            assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-9), \
                f"{cfg.name}: расходится {col} — lookahead"
        print(f"  ok  {cfg.name}: {len(a)} сделок совпали бит в бит")


def test_entry_not_before_next_bar():
    """Вход не может произойти на баре, где возник сигнал."""
    df = _with_ref(make_ohlcv(n=6000, regime="edge", seed=5))
    for d in CONFIGS:
        cfg = StrategyConfig.from_dict(dict(d))
        r = run_backtest(df, cfg, "T")
        if not r.n_trades:
            continue
        # цена входа обязана лежать внутри диапазона своего бара
        o, h, l = (df[x].to_numpy() for x in ("open", "high", "low"))
        eb = r.trades["entry_bar"].to_numpy()
        ep = r.trades["entry_price"].to_numpy()
        assert ((ep >= l[eb] * (1 - 1e-9)) & (ep <= h[eb] * (1 + 1e-9))).all(), \
            f"{cfg.name}: цена входа вне диапазона бара входа"
        assert (r.trades["exit_bar"].to_numpy() >= eb).all(), \
            f"{cfg.name}: выход раньше входа"
        print(f"  ok  {cfg.name}: входы внутри своих баров")


def test_htf_trend_is_causal():
    """Фильтр старшего ТФ не должен зависеть от будущих баров.

    Тихий и очень частый баг: ресемпл в H4 без сдвига даёт значение старшего
    бара уже на его первом младшем баре, то есть за 3 часа до его закрытия.
    """
    df = make_ohlcv(n=3000, regime="edge", seed=9)
    t_full = ind.htf_trend_causal(df, 4, 50)
    for cut in (1500, 1777, 2001):
        t_cut = ind.htf_trend_causal(df.iloc[:cut].reset_index(drop=True), 4, 50)
        assert (t_full[:cut] == t_cut).all(), \
            f"htf_trend зависит от будущего (обрезка на {cut})"
    print("  ok  htf_trend causal на всех обрезках")


def test_costs_floor_enforced():
    """Конфиг с издержками ниже 0.3% round-trip не должен создаваться."""
    base = {"name": "x", "entry": {"type": "tsmom", "params": {}},
            "stop": {"type": "atr", "params": {}}, "exit": {"type": "time", "params": {}}}
    try:
        StrategyConfig.from_dict({**base, "costs": {"fee_bps_per_side": 1.0,
                                                    "slip_bps_per_side": 1.0}})
    except ConfigError:
        print("  ok  заниженные издержки отклонены")
        return
    raise AssertionError("конфиг с издержками 4 bps round-trip прошёл — дыра в защите")


def test_costs_reduce_result():
    """Больше издержек — строго хуже результат. Иначе они где-то не применяются."""
    df = make_ohlcv(n=8000, regime="edge", seed=21)
    base = {"name": "c", "entry": {"type": "donchian_breakout", "params": {"period": 20}},
            "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
            "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 100}}}
    lo = run_backtest(df, StrategyConfig.from_dict(dict(base)), "T")
    hi_cfg = dict(base)
    hi_cfg["costs"] = {"fee_bps_per_side": 30.0, "slip_bps_per_side": 30.0}
    hi = run_backtest(df, StrategyConfig.from_dict(hi_cfg), "T")
    assert hi.R.mean() < lo.R.mean(), "рост издержек не ухудшил результат"
    print(f"  ok  издержки работают: {lo.R.mean():+.3f}R -> {hi.R.mean():+.3f}R")


def test_tiny_stop_is_punished():
    """Микроскопический стоп обязан быть дороже в R, а не выгоднее.

    Проверяет ключевое свойство учёта: издержки считаются от НОТИОНАЛА,
    а нотионал на единицу риска взрывается при крошечном стопе.
    """
    df = make_ohlcv(n=8000, regime="edge", seed=33)
    base = {"name": "s", "entry": {"type": "donchian_breakout", "params": {"period": 20}},
            "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 100}}}
    wide = run_backtest(df, StrategyConfig.from_dict(
        {**base, "stop": {"type": "pct", "params": {"pct": 0.02}}}), "T")
    tiny = run_backtest(df, StrategyConfig.from_dict(
        {**base, "stop": {"type": "pct", "params": {"pct": 0.0015}}}), "T")
    cost_wide = wide.trades["fees"].sum() / wide.trades["risk"].sum()
    cost_tiny = tiny.trades["fees"].sum() / tiny.trades["risk"].sum()
    assert cost_tiny > cost_wide * 5, \
        f"мелкий стоп не наказан издержками: {cost_tiny:.3f}R vs {cost_wide:.3f}R"
    print(f"  ok  издержки на единицу риска: стоп 2% -> {cost_wide:.3f}R, "
          f"стоп 0.15% -> {cost_tiny:.3f}R")


def test_stop_on_losing_side():
    """Стоп обязан быть на проигрышной стороне входа — во ВСЕХ сделках.

    Ловит целый класс поломок: структурный стоп (канал/свинг/край зоны) может
    оказаться по другую сторону цены, и тогда abs() превращает «стоп-лосс»
    в прибыльный выход. Симптом — winrate под 97% и PF за 20.
    """
    df = make_ohlcv(n=12000, regime="edge", seed=41)
    combos = [
        ("bollinger_meanrev", {"type": "channel", "params": {"period": 20}}),
        ("bollinger_meanrev", {"type": "structure", "params": {"lookback": 20}}),
        ("donchian_breakout", {"type": "channel", "params": {"period": 20}}),
        ("rsi_threshold", {"type": "structure", "params": {"lookback": 20}}),
        ("order_block", {"type": "block", "params": {}}),
    ]
    for entry, stop in combos:
        cfg = StrategyConfig.from_dict(
            {"name": f"{entry}/{stop['type']}", "entry": {"type": entry, "params": {}},
             "stop": stop, "exit": {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 96}}})
        r = run_backtest(df, cfg, "T")
        if not r.n_trades:
            continue
        t = r.trades
        d = t["direction"].to_numpy()
        bad = ((d > 0) & (t["stop_price"].to_numpy() >= t["entry_price"].to_numpy())) | \
              ((d < 0) & (t["stop_price"].to_numpy() <= t["entry_price"].to_numpy()))
        assert not bad.any(), (
            f"{cfg.name}: {bad.sum()} сделок с перевёрнутым стопом — "
            f"«стоп» на выигрышной стороне даёт фантомную прибыль")
        # и никаких прибыльных стоп-лоссов
        st = t[t["exit_reason"] == "stop"]
        assert (st["R"] <= 1e-9).all(), \
            f"{cfg.name}: есть стоп-лоссы с положительным R"
        print(f"  ok  {cfg.name}: {len(t)} сделок, стопы на верной стороне")


def test_funding_accrual():
    """Funding начисляется по числу пройденных 8-часовых границ."""
    step = 8 * 3_600_000
    t0 = 0
    assert _funding_cost(None, 0.0001, t0, t0 + 1000, 100_000, 1) == 0.0
    one = _funding_cost(None, 0.0001, t0, t0 + step, 100_000, 1)
    three = _funding_cost(None, 0.0001, t0, t0 + 3 * step, 100_000, 1)
    assert abs(one - 10.0) < 1e-6, one
    assert abs(three - 30.0) < 1e-6, three
    # шорт получает то, что платит лонг
    assert _funding_cost(None, 0.0001, t0, t0 + step, 100_000, -1) == -one
    print("  ok  funding: 8ч->10, 24ч->30, шорт зеркален")


def test_synthetic_noise_has_no_edge():
    """На чистом шуме честная стратегия обязана быть в минусе после издержек.
    Если она в плюсе — сломан либо генератор, либо учёт издержек."""
    cfg = StrategyConfig.from_dict(
        {"name": "n", "entry": {"type": "donchian_breakout", "params": {"period": 20}},
         "stop": {"type": "atr", "params": {"period": 14, "mult": 2.0}},
         "exit": {"type": "trailing_atr", "params": {"period": 14, "mult": 3.0,
                                                     "max_bars": 200}}})
    exps = []
    for seed in (1, 2, 3):
        df = make_ohlcv(n=20000, regime="noise", seed=seed)
        exps.append(run_backtest(df, cfg, "T", funding=make_funding(df)).R.mean())
    assert np.mean(exps) < 0, f"на шуме получилась положительная ожидаемость: {exps}"
    print(f"  ok  шум даёт минус: {[round(e,3) for e in exps]}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{'='*60}\n{len(tests)-failed}/{len(tests)} тестов пройдено")
    sys.exit(1 if failed else 0)
