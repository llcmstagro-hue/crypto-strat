"""
Проверка по РЕЖИМАМ РЫНКА — приёмочное требование поверх барьеров р.4.

Зачем отдельно от walk-forward. Walk-forward отвечает «держится ли эдж в
большинстве окон», но окна нарезаны механически по календарю и режимы в них
перемешаны. Здесь вопрос конкретнее и жёстче: **стратегия зарабатывала в
медвежке или только в росте?**

Это принципиально под CFT. Стратегия, которая делает всю прибыль в бычьем
2021 и 2023-24, а в 2018/2022 отдаёт — это не эдж, это бета к рынку с
задержкой. На челлендже она встретит медвежий период и выбьет лимит.

Критерий «достойной» стратегии:
  * положительная ожидаемость МИНИМУМ в одном медвежьем и одном бычьем режиме;
  * ни в одном режиме с достаточной выборкой нет катастрофы;
  * прибыль не сконцентрирована в одном режиме (доля лучшего режима в общем
    результате не должна быть подавляющей).

Границы режимов заданы по факту движения корзины, а НЕ подобраны под
результат стратегии — иначе это была бы очередная подгонка.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.metrics import trade_metrics

# Режимы крипторынка по календарю. Определены по динамике мейджоров
# (см. годовую таблицу в PROGRESS.md), ДО и НЕЗАВИСИМО от любых стратегий.
REGIMES = [
    ("бык 2017",        "2017-08-01", "2018-01-15", "bull"),
    ("медвежка 2018",   "2018-01-15", "2019-01-01", "bear"),
    ("восстановл. 2019","2019-01-01", "2020-03-01", "bull"),
    ("бык 2020-21",     "2020-03-01", "2021-11-10", "bull"),
    ("медвежка 2022",   "2021-11-10", "2023-01-01", "bear"),
    ("бык 2023-24",     "2023-01-01", "2025-01-01", "bull"),
    ("боковик 2025",    "2025-01-01", "2025-12-01", "sideways"),
    ("медвежка 2026",   "2025-12-01", "2026-08-01", "bear"),
]


def _ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def regime_breakdown(results: list, min_trades: int = 20) -> dict:
    """Разбор сделок по режимам. results — список BacktestResult по корзине."""
    rows = []
    for name, a, b, kind in REGIMES:
        ta, tb = _ms(a), _ms(b)
        R, pnl = [], []
        for r in results:
            if not r.n_trades:
                continue
            t = r.trades
            m = (t["entry_ts"] >= ta) & (t["entry_ts"] < tb)
            if m.any():
                R.append(t.loc[m, "R"].to_numpy())
                pnl.append(t.loc[m, "net_pnl"].to_numpy())
        allR = np.concatenate(R) if R else np.array([])
        mm = trade_metrics(allR)
        rows.append({
            "режим": name, "тип": kind,
            "с": a[:7], "по": b[:7],
            "сделок": mm["n_trades"],
            "expectancy_R": round(mm["expectancy_R"], 4),
            "PF": round(mm["profit_factor"], 3),
            "winrate": round(mm["winrate"], 3),
            "total_R": round(mm["total_R"], 1),
            "достаточно": mm["n_trades"] >= min_trades,
        })
    return {"rows": rows}


def regime_verdict(breakdown: dict, min_trades: int = 20,
                   catastrophe_R: float = -0.25,
                   max_concentration: float = 0.80) -> dict:
    """Приговор по режимам. Возвращает пройдено/нет + разбор причин."""
    rows = [r for r in breakdown["rows"] if r["достаточно"]]
    if not rows:
        return {"passed": False, "note": "ни в одном режиме нет достаточной выборки",
                "checks": {}}

    bulls = [r for r in rows if r["тип"] == "bull"]
    bears = [r for r in rows if r["тип"] == "bear"]
    bull_ok = any(r["expectancy_R"] > 0 for r in bulls)
    bear_ok = any(r["expectancy_R"] > 0 for r in bears)
    no_catastrophe = all(r["expectancy_R"] > catastrophe_R for r in rows)

    # концентрация прибыли: сколько даёт лучший режим от суммы положительных
    pos = [r["total_R"] for r in rows if r["total_R"] > 0]
    concentration = (max(pos) / sum(pos)) if pos else 1.0
    diversified = concentration <= max_concentration

    checks = {
        "в плюсе хотя бы в одном БЫЧЬЕМ режиме": bull_ok,
        "в плюсе хотя бы в одном МЕДВЕЖЬЕМ режиме": bear_ok,
        "нет катастрофы ни в одном режиме": no_catastrophe,
        "прибыль не сконцентрирована в одном режиме": diversified,
    }
    passed = all(checks.values())

    bear_names = [r["режим"] for r in bears if r["expectancy_R"] > 0]
    bull_names = [r["режим"] for r in bulls if r["expectancy_R"] > 0]
    note = (f"режимов с выборкой {len(rows)}/{len(breakdown['rows'])}; "
            f"в плюсе быки: {', '.join(bull_names) or '—'}; "
            f"медвежки: {', '.join(bear_names) or '—'}; "
            f"концентрация прибыли {concentration*100:.0f}%")
    return {"passed": passed, "checks": checks, "note": note,
            "concentration": concentration}


# --------------------------------------------------------------------------- #
# СВЕЖЕСТЬ ЭДЖА — обязательное условие достойности, наравне с 7 барьерами
# --------------------------------------------------------------------------- #
FRESH_FROM = "2024-01-01"      # «последний режим»
RECENT_FROM = "2025-01-01"     # ближний хвост


def freshness_check(results: list, min_trades: int = 30) -> dict:
    """Жив ли эдж СЕЙЧАС, а не только в истории.

    Главный урок прогона по пробою: стратегия может иметь безупречную
    девятилетнюю статистику и при этом быть мёртвой последние два года.
    Исторический эдж не даёт права на форвард — рынок уже другой.

    Требуется положительная ожидаемость И в окне 2024+, И в окне 2025+.
    Второе окно уже и шумнее, но именно оно отвечает на вопрос «работает ли
    прямо сейчас»; первое даёт достаточную выборку, чтобы это не было
    гаданием по десяти сделкам.

    Провал = вердикт `historical-only edge`. Это НЕ смягчение и НЕ ужесточение
    порогов барьеров — это отдельное требование поверх них.
    """
    out = {}
    for key, start in (("fresh_2024", FRESH_FROM), ("recent_2025", RECENT_FROM)):
        ts0 = _ms(start)
        R = []
        for r in results:
            if not r.n_trades:
                continue
            t = r.trades
            m = t["entry_ts"] >= ts0
            if m.any():
                R.append(t.loc[m, "R"].to_numpy())
        arr = np.concatenate(R) if R else np.array([])
        mm = trade_metrics(arr)
        out[key] = {"from": start, "n": mm["n_trades"],
                    "expectancy_R": round(mm["expectancy_R"], 4),
                    "profit_factor": round(mm["profit_factor"], 3),
                    "total_R": round(mm["total_R"], 1)}

    f, rc = out["fresh_2024"], out["recent_2025"]
    checks = {
        f"выборка с {FRESH_FROM} >= {min_trades}": f["n"] >= min_trades,
        f"ожидаемость с {FRESH_FROM} > 0": f["expectancy_R"] > 0,
        f"ожидаемость с {RECENT_FROM} > 0": rc["expectancy_R"] > 0,
    }
    passed = all(checks.values())
    note = (f"с {FRESH_FROM}: n={f['n']} exp={f['expectancy_R']:+.3f}R "
            f"PF={f['profit_factor']:.2f} | с {RECENT_FROM}: n={rc['n']} "
            f"exp={rc['expectancy_R']:+.3f}R PF={rc['profit_factor']:.2f}")
    return {"passed": passed, "checks": checks, "note": note, "windows": out,
            "verdict": "ok" if passed else "historical-only edge"}


def format_regimes(breakdown: dict) -> str:
    L = [f"{'режим':<20}{'тип':<10}{'сделок':>8}{'exp,R':>9}{'PF':>7}{'итог,R':>9}"]
    L.append("-" * 63)
    for r in breakdown["rows"]:
        mark = "" if r["достаточно"] else "  (мало сделок)"
        L.append(f"{r['режим']:<20}{r['тип']:<10}{r['сделок']:>8}"
                 f"{r['expectancy_R']:>+9.3f}{r['PF']:>7.2f}{r['total_R']:>+9.1f}{mark}")
    return "\n".join(L)
