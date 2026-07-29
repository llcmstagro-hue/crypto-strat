"""
Модуль [6] — бумажный форвард и монитор деградации (р.6.6).

Что это и чем НЕ является.

Это НАБЛЮДЕНИЕ, а не торговля. Стратегия сюда попадает не потому, что прошла
фильтр (она не прошла), а потому что её нужно РАЗЛИЧИТЬ: эдж умер или эдж спит.
Бэктест на это ответить не может — он смотрит в прошлое, где эдж был. Ответ
даёт только поведение на данных, которых стратегия не видела ни в каком виде.

Боевого допуска наблюдение НЕ выдаёт. Оно копит доказательства.

⛔ ПАРАМЕТРЫ ЗАМОРОЖЕНЫ НАВСЕГДА (р.6.6). Никакой переоптимизации на свежих
данных: система учится ОТБОРУ, а не параметрам. Если конфиг подкрутить по
ходу наблюдения, наблюдение перестанет что-либо доказывать — оно превратится
в подгонку в динамике, идеально объясняющую вчера и проваливающую завтра.

ПОРОГИ ДЕГРАДАЦИИ — ИЗ МОНТЕ-КАРЛО, НЕ ИЗ ГОЛОВЫ (р.6.6). Максимальная серия
убытков в бэктесте — это НЕ потолок, а лишь то, что уместилось в выборку. Чем
дольше торгуешь, тем длиннее серии увидишь даже при неизменном эдже. Поэтому
жёлтый и красный пороги считаются как перцентили распределения просадок,
полученного бутстрепом из baseline-распределения R данной стратегии.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# МОНТЕ-КАРЛО BASELINE
# --------------------------------------------------------------------------- #
HORIZONS = (25, 50, 100, 200, 300, 500, 750, 1000)


def monte_carlo_baseline(R: np.ndarray, n_paths: int = 20_000,
                         horizon: int | None = None, seed: int = 12345,
                         rolling_n: int = 40) -> dict:
    """Распределение просадок и серий убытков при НЕИЗМЕННОМ эдже.

    Бутстреп с возвращением из наблюдённого распределения R. Порядок сделок
    перемешивается — именно он и создаёт серии, а мы хотим знать, какие серии
    НОРМАЛЬНЫ для этой стратегии, а какие аномальны.

    Пример из брифа: winrate 57% -> серия из 5 убытков имеет вероятность
    0.43^5 ~ 1.5%, то есть на 200 сделках встретится ~3 раза ПО ТЕОРИИ. Порог
    «5 подряд — выключаем» резал бы рабочий эдж трижды за цикл.
    """
    R = np.asarray(R, dtype=float)
    if len(R) < 30:
        return {"error": "слишком мало сделок для Монте-Карло", "n": len(R)}
    rng = np.random.default_rng(seed)
    H = int(horizon or len(R))

    draws = rng.choice(R, size=(n_paths, H), replace=True)
    eq = np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (peak - eq).max(axis=1)

    losses = draws < 0
    streaks = np.zeros(n_paths, dtype=int)
    cur = np.zeros(n_paths, dtype=int)
    for j in range(H):
        cur = np.where(losses[:, j], cur + 1, 0)
        streaks = np.maximum(streaks, cur)

    q = lambda a, p: float(np.percentile(a, p))

    # --- ПОРОГИ КАК ФУНКЦИЯ ГОРИЗОНТА ---
    # Порог просадки нельзя задать одним числом. Просадка растёт с числом
    # сделок ДАЖЕ ПРИ НЕИЗМЕННОМ ЭДЖЕ — ровно та же ошибка, что и «максимум
    # серии убытков в бэктесте +1» из р.6.6, только про просадки. Порог,
    # посчитанный на 300 сделок, ложно сработает на 900-й.
    curve = {}
    for H_i in HORIZONS:
        d_i = rng.choice(R, size=(min(n_paths, 8000), H_i), replace=True)
        e_i = np.cumsum(d_i, axis=1)
        dd_i = (np.maximum.accumulate(e_i, axis=1) - e_i).max(axis=1)
        curve[H_i] = {"p95": q(dd_i, 95), "p99": q(dd_i, 99), "p50": q(dd_i, 50)}

    # --- КАЛИБРОВКА КАТЯЩЕГОСЯ ОЖИДАНИЯ ---
    # «Ожидание ушло в минус» — плохой триггер сам по себе: на здоровой
    # стратегии скользящее среднее по 30-40 сделкам уходит в минус треть
    # времени. Нужен ПЕРЦЕНТИЛЬ этого распределения, а не знак.
    roll_paths = rng.choice(R, size=(4000, max(rolling_n * 4, 200)), replace=True)
    rm = np.lib.stride_tricks.sliding_window_view(
        roll_paths, rolling_n, axis=1).mean(axis=2)
    roll_min = rm.min(axis=1)          # худшее окно на пути
    roll_any = rm.reshape(-1)

    return {
        "n_trades_baseline": int(len(R)),
        "horizon": H,
        "rolling_n": rolling_n,
        "expectancy_R": float(R.mean()),
        "winrate": float((R > 0).mean()),
        "dd_R": {"p50": q(dd, 50), "p90": q(dd, 90), "p95": q(dd, 95),
                 "p99": q(dd, 99), "max": float(dd.max())},
        "dd_curve": curve,
        "loss_streak": {"p50": q(streaks, 50), "p95": q(streaks, 95),
                        "p99": q(streaks, 99), "max": int(streaks.max())},
        "rolling": {
            # доля времени, когда катящееся ожидание отрицательно У ЗДОРОВОЙ
            # стратегии — если она велика, «минус» вообще не сигнал
            "share_negative": float((roll_any < 0).mean()),
            "p05_any_window": q(roll_any, 5),
            "p05_worst_window": q(roll_min, 5),
            "p01_worst_window": q(roll_min, 1),
        },
        "yellow_dd_R": q(dd, 95),
        "red_dd_R": q(dd, 99),
        "normal_loss_streak_p95": q(streaks, 95),
    }


PLANNED_HORIZON = 300     # на сколько сделок вперёд объявляется наблюдение


def thresholds_at(mc: dict, n: int) -> tuple[float, float]:
    """Пороги просадки для ПЛАНИРУЕМОГО горизонта наблюдения.

    Тонкость, на которой легко ошибиться (я ошибся при первой реализации).
    Просадка — величина монотонно НЕубывающая: `dd(k)` только растёт по ходу
    наблюдения. Поэтому её максимум за весь путь равен её значению в конце,
    и сравнение ТЕКУЩЕЙ просадки с перцентилем, посчитанным на ВЕСЬ горизонт,
    даёт ровно заявленную частоту ложных тревог: 5% для жёлтого, 1% для красного.

    А вот «порог, соответствующий текущему числу сделок» — ошибка: он
    пересчитывается и проверяется на КАЖДОЙ сделке, то есть тест проводится
    сотни раз, и фактическая вероятность хоть раз пробить его стремится к
    единице. В реплее это дало красный на 21-й сделке из 502 у стратегии,
    которая потом отработала весь период в плюс.

    Поэтому горизонт фиксируется заранее (`PLANNED_HORIZON`) и растягивается
    только если наблюдение фактически оказалось длиннее объявленного.
    """
    n = max(int(n or 0), PLANNED_HORIZON)
    curve = mc.get("dd_curve") or {}
    if not curve:
        return mc.get("yellow_dd_R", np.inf), mc.get("red_dd_R", np.inf)
    hs = sorted(int(h) for h in curve)
    if n <= hs[0]:
        c = curve[str(hs[0])] if str(hs[0]) in curve else curve[hs[0]]
        return c["p95"], c["p99"]
    if n >= hs[-1]:
        c = curve[str(hs[-1])] if str(hs[-1]) in curve else curve[hs[-1]]
        return c["p95"], c["p99"]
    for a, b in zip(hs, hs[1:]):
        if a <= n <= b:
            ca = curve[str(a)] if str(a) in curve else curve[a]
            cb = curve[str(b)] if str(b) in curve else curve[b]
            w = (n - a) / (b - a)
            return (ca["p95"] + w * (cb["p95"] - ca["p95"]),
                    ca["p99"] + w * (cb["p99"] - ca["p99"]))
    return mc.get("yellow_dd_R", np.inf), mc.get("red_dd_R", np.inf)


# --------------------------------------------------------------------------- #
# ЗАМОРОЗКА
# --------------------------------------------------------------------------- #
@dataclass
class FrozenStrategy:
    name: str
    config: dict
    tf: str
    symbols: list
    frozen_at: str = field(default_factory=_now)
    frozen_data_end: str = ""
    status: str = "paper-forward"        # НЕ боевой
    admission: str = "НЕ ПРОШЛА ФИЛЬТР — наблюдение без денег"
    why_watching: str = ""
    baseline: dict = field(default_factory=dict)
    monte_carlo: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def load(path: str) -> "FrozenStrategy":
        with open(path, encoding="utf-8") as f:
            return FrozenStrategy(**json.load(f))


# --------------------------------------------------------------------------- #
# ОЦЕНКА СОСТОЯНИЯ
# --------------------------------------------------------------------------- #
def evaluate(frozen: FrozenStrategy, trades: pd.DataFrame,
             rolling_n: int = 40) -> dict:
    """Светофор р.6.6 по сделкам, случившимся ПОСЛЕ заморозки.

      🟢 зелёный  — в пределах нормального разброса
      🟡 жёлтый   — просадка прошла 95-й перцентиль Монте-Карло ЛИБО катящееся
                    ожидание отрицательно на выборке >= 30 сделок
      🔴 красный  — просадка пробила 99-й перцентиль ЛИБО ожидание устойчиво
                    отрицательно на >= 50 сделках

    Серия убытков НЕ является триггером сама по себе: длинные серии нормальны
    при живом эдже. Она приводится справочно, рядом с тем, что Монте-Карло
    считает нормой для ЭТОЙ стратегии.
    """
    mc = frozen.monte_carlo or {}
    if trades is None or not len(trades):
        return {"status": "🟢 нет данных", "n": 0,
                "note": "после заморозки сделок ещё не было — наблюдение начато, "
                        "ждём новые бары"}

    R = trades["R"].to_numpy(float)
    eq = np.cumsum(R)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    roll = float(R[-rolling_n:].mean()) if len(R) >= rolling_n else float(R.mean())

    streak = best = 0
    for r in R:
        streak = streak + 1 if r < 0 else 0
        best = max(best, streak)

    # пороги, СООТВЕТСТВУЮЩИЕ числу уже наблюдённых сделок
    yellow, red = thresholds_at(mc, len(R))
    rollq = mc.get("rolling") or {}
    roll_yellow = rollq.get("p05_any_window", 0.0)
    roll_red = rollq.get("p01_worst_window", rollq.get("p05_worst_window", 0.0))

    reasons = []
    status = "🟢 зелёный"
    y_dd = dd >= yellow
    # НЕ «ожидание ушло в минус» (у здоровой стратегии так треть времени),
    # а «ожидание ниже 5-го перцентиля собственного распределения»
    y_roll = len(R) >= rolling_n and roll <= roll_yellow
    if y_dd or y_roll:
        status = "🟡 ЖЁЛТЫЙ — сайз вдвое вниз"
        if y_dd:
            reasons.append(f"просадка {dd:.1f}R прошла 95-й перцентиль "
                           f"Монте-Карло на горизонт наблюдения ({yellow:.1f}R)")
        if y_roll:
            reasons.append(f"катящееся ожидание {roll:+.3f}R ниже 5-го перцентиля "
                           f"нормы ({roll_yellow:+.3f}R)")

    r_dd = dd >= red
    r_roll = len(R) >= max(50, rolling_n) and roll <= roll_red
    if r_dd or r_roll:
        status = "🔴 КРАСНЫЙ — стоп и карантин"
        if r_dd:
            reasons.append(f"просадка {dd:.1f}R пробила 99-й перцентиль "
                           f"на горизонт наблюдения ({red:.1f}R)")
        if r_roll:
            reasons.append(f"катящееся ожидание {roll:+.3f}R ниже 1-го перцентиля "
                           f"нормы ({roll_red:+.3f}R)")

    base_exp = (frozen.baseline or {}).get("expectancy_R", 0.0)
    return {
        "status": status,
        "n": len(R),
        "expectancy_R": float(R.mean()),
        "baseline_expectancy_R": base_exp,
        "rolling_expectancy_R": roll,
        "drawdown_R": dd,
        "max_loss_streak": best,
        "normal_streak_p95": mc.get("normal_loss_streak_p95"),
        "yellow_dd_R": yellow, "red_dd_R": red,
        "reasons": reasons,
        "note": ("серия убытков сама по себе не триггер: при живом эдже "
                 f"нормой считается до {mc.get('normal_loss_streak_p95', '?')} подряд"),
    }


def format_status(frozen: FrozenStrategy, ev: dict) -> str:
    L = ["=" * 78,
         f"БУМАЖНЫЙ ФОРВАРД: {frozen.name}",
         "=" * 78,
         f"  статус допуска : {frozen.admission}",
         f"  заморожен      : {frozen.frozen_at} (данные по {frozen.frozen_data_end})",
         f"  ТФ / корзина   : {frozen.tf} / {', '.join(frozen.symbols)}",
         f"  зачем наблюдаем: {frozen.why_watching}",
         "-" * 78]
    b, mc = frozen.baseline, frozen.monte_carlo
    if b:
        L.append(f"  BASELINE (валидация): сделок {b.get('n_trades')}, "
                 f"exp {b.get('expectancy_R', 0):+.3f}R, "
                 f"winrate {b.get('winrate', 0)*100:.1f}%, PF {b.get('profit_factor', 0):.2f}")
    if mc and "dd_R" in mc:
        L.append(f"  МОНТЕ-КАРЛО (20 000 путей, бутстреп из собственного R):")
        cur = mc.get("dd_curve") or {}
        if cur:
            L.append("     порог просадки РАСТЁТ с числом сделок:")
            for h in sorted(int(x) for x in cur):
                c = cur.get(str(h), cur.get(h))
                L.append(f"        {h:>4} сделок: 🟡 {c['p95']:>5.1f}R  "
                         f"🔴 {c['p99']:>5.1f}R  (медиана {c['p50']:.1f}R)")
        L.append(f"     серия убытков: медиана {mc['loss_streak']['p50']:.0f} | "
                 f"95% {mc['loss_streak']['p95']:.0f} | 99% {mc['loss_streak']['p99']:.0f}")
        rl = mc.get("rolling") or {}
        if rl:
            L.append(f"     катящееся ожидание по {mc.get('rolling_n')} сделкам "
                     f"у ЗДОРОВОЙ стратегии отрицательно "
                     f"{rl['share_negative']*100:.0f}% времени")
            L.append(f"        -> «ушло в минус» НЕ триггер; порог 🟡 "
                     f"{rl['p05_any_window']:+.3f}R, 🔴 {rl['p01_worst_window']:+.3f}R")
    L.append("-" * 78)
    L.append(f"  СОСТОЯНИЕ: {ev['status']}   (сделок после заморозки: {ev['n']})")
    if ev["n"]:
        L.append(f"     ожидаемость {ev['expectancy_R']:+.3f}R против baseline "
                 f"{ev['baseline_expectancy_R']:+.3f}R")
        L.append(f"     просадка {ev['drawdown_R']:.1f}R "
                 f"(пороги на горизонт наблюдения: 🟡 {ev['yellow_dd_R']:.1f}R / "
                 f"🔴 {ev['red_dd_R']:.1f}R)")
        L.append(f"     макс. серия убытков {ev['max_loss_streak']} "
                 f"(норма до {ev['normal_streak_p95']})")
    for r in ev.get("reasons", []):
        L.append(f"     • {r}")
    L.append(f"  {ev['note']}")
    L.append("=" * 78)
    return "\n".join(L)
