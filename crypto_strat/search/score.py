"""
Research Score и Critic (р.6.7) — дешёвый гейт ПЕРЕД дорогим бэктестом.

Разделение ролей, которое важно не размывать:
  * Research Score оценивает КАЧЕСТВО ИДЕИ (логика, формализуемость, сложность,
    новизна). Это НЕ проверка эджа и НЕ замена фильтру р.4 — идея с 95 баллами
    точно так же обязана пройти все семь барьеров.
  * Critic — отдельная роль «адвокат дьявола»: ищет, ПОЧЕМУ идея НЕ сработает,
    и делает это ДО того, как на неё потрачены минуты счёта.

Почему гейт стоит до бэктеста, а не после: каждая протестированная гипотеза
поднимает планку барьера 5 ДЛЯ ВСЕХ ОСТАЛЬНЫХ. Прогнать заведомо мусорную
идею — это не «просто потратить время», это ухудшить условия хорошим
кандидатам. Поэтому мусор обязан отсеиваться ДЕШЁВО и НЕ ПОПАДАТЬ в счётчик.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.blocks import ENTRY_BLOCKS, FILTER_BLOCKS
from ..engine.config import StrategyConfig

SCORE_THRESHOLD = 70

# Типы стратегий (роль Classifier из р.6.7)
BLOCK_TYPE = {
    "donchian_breakout": "breakout",
    "keltner_breakout": "breakout",
    "tsmom": "momentum",
    "ma_cross": "trend",
    "bollinger_meanrev": "mean-reversion",
    "rsi_threshold": "mean-reversion",
    "rsi_divergence": "mean-reversion",
    "order_block": "SMC",
    "fvg": "SMC",
    "level_retest": "SMC",
}

# Блоки, требующие объёма
VOLUME_BLOCKS = {"volume"}


def classify(cfg: StrategyConfig) -> str:
    return BLOCK_TYPE.get(cfg.entry["type"], "other")


# --------------------------------------------------------------------------- #
# RESEARCH SCORE
# --------------------------------------------------------------------------- #
@dataclass
class Idea:
    """Метаданные идеи, которых нет в конфиге, но которые нужны для оценки."""
    rationale: str = ""            # рыночная логика: ПОЧЕМУ это должно работать
    sources: list = field(default_factory=list)
    cross_market: bool = False     # механизм не завязан на конкретный рынок


def research_score(cfg: StrategyConfig, idea: Idea, grid: dict,
                   is_duplicate: bool) -> tuple[int, dict]:
    """0..100 по весам р.6.7. Возвращает (балл, разбивка)."""
    b: dict[str, int] = {}

    # понятная рыночная логика — 20
    r = (idea.rationale or "").strip()
    b["рыночная логика"] = 20 if len(r) >= 80 else (12 if len(r) >= 30 else 0)

    # формализуется без двусмысленностей — 15
    # конфиг собран из блоков и исполняется движком, значит формализован
    # по построению; штрафуем только за неизвестные блоки
    ok_blocks = (cfg.entry["type"] in ENTRY_BLOCKS
                 and all(f["type"] in FILTER_BLOCKS for f in cfg.filters))
    b["формализуемость"] = 15 if ok_blocks else 0

    # нет look-ahead / repaint — 15
    # все блоки causal by construction и покрыты тестом на обрезку ряда
    b["без look-ahead"] = 15

    # подтверждена несколькими источниками — 10
    n_src = len([s for s in idea.sources if s])
    b["источники"] = 10 if n_src >= 2 else (5 if n_src == 1 else 0)

    # потенциал на разных рынках — 15
    b["кросс-рыночность"] = 15 if idea.cross_market else 7

    # разумная сложность — 15
    # штраф и за число правил, и за ширину сетки: широкий перебор — это
    # тоже сложность, просто спрятанная в пространство параметров
    n_rules = cfg.n_rules()
    n_combos = 1
    for v in grid.values():
        n_combos *= max(len(v), 1)
    pen = 0
    if n_rules > 6:
        pen += 8
    elif n_rules > 4:
        pen += 3
    if n_combos > 36:
        pen += 7
    elif n_combos > 24:
        pen += 3
    b["сложность"] = max(0, 15 - pen)

    # не дубликат — 10
    b["новизна"] = 0 if is_duplicate else 10

    return sum(b.values()), b


# --------------------------------------------------------------------------- #
# CRITIC — ищет, почему НЕ сработает
# --------------------------------------------------------------------------- #
@dataclass
class CriticVerdict:
    verdict: str                       # "pass" | "reject"
    objections: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def critic(cfg: StrategyConfig, grid: dict, idea: Idea, *,
           has_volume: bool = True, is_duplicate: bool = False,
           typical_atr_pct: float = 0.02) -> CriticVerdict:
    """Адвокат дьявола. `reject` = не пускать в дорогой тест вообще.

    Возражения, которые НЕ являются reject, всё равно сохраняются в базу:
    если гипотеза потом провалится, видно, предупреждал ли Critic заранее.
    """
    obj: list[str] = []
    hard = False

    if is_duplicate:
        obj.append("дубликат уже протестированной логики (по отпечатку конфига)")
        hard = True

    # объёмные блоки на данных без объёма
    used_filters = {f["type"] for f in cfg.filters}
    if not has_volume and (used_filters & VOLUME_BLOCKS):
        obj.append("использует объёмный фильтр, а в данных объёма нет")
        hard = True

    # много правил на выборке — прямой красный флаг р.4
    if cfg.n_rules() > 6:
        obj.append(f"правил {cfg.n_rules()} (>6): много условий, выборка будет срезана")
        hard = True

    # ширина перебора
    n_combos = 1
    for v in grid.values():
        n_combos *= max(len(v), 1)
    if n_combos > 40:
        obj.append(f"сетка {n_combos} комбинаций — широкий перебор поднимает "
                   f"планку барьера 5 всем кандидатам")
        hard = True

    # подсвечный стоп: фиксированный процент заметно меньше типичного ATR
    if cfg.stop["type"] == "pct":
        pct = float(cfg.stop.get("params", {}).get("pct", 0.01))
        if pct < 0.5 * typical_atr_pct:
            obj.append(f"стоп {pct*100:.2f}% при типичном ATR {typical_atr_pct*100:.1f}% — "
                       f"подсвечный стоп честно не тестируется (р.10 п.3)")
            hard = True

    # односторонняя торговля на истории с несколькими медвежками
    if cfg.direction in ("long", "short"):
        obj.append(f"только {cfg.direction}: результат будет зависеть от режима рынка, "
                   f"а история содержит и быки, и медвежки")

    # нет рыночной логики
    if len((idea.rationale or "").strip()) < 30:
        obj.append("не сформулирована рыночная логика — почему это должно работать")

    # экзотика: выход по времени без стопа по структуре на трендовом входе
    if cfg.exit["type"] == "time" and cfg.entry["type"] in ("donchian_breakout",
                                                            "keltner_breakout"):
        obj.append("пробойный вход с выходом строго по времени: обрезает "
                   "правый хвост, ради которого пробой и берут")

    return CriticVerdict("reject" if hard else "pass", obj)


def gate(cfg: StrategyConfig, grid: dict, idea: Idea, *, has_volume: bool,
         is_duplicate: bool, threshold: int = SCORE_THRESHOLD) -> dict:
    """Полный гейт: Score + Critic. Решение о допуске к бэктесту."""
    score, breakdown = research_score(cfg, idea, grid, is_duplicate)
    cv = critic(cfg, grid, idea, has_volume=has_volume, is_duplicate=is_duplicate)
    admitted = cv.passed and score >= threshold
    if not admitted:
        why = []
        if not cv.passed:
            why.append("Critic: " + "; ".join(cv.objections[:2]))
        if score < threshold:
            why.append(f"Research Score {score} < {threshold}")
        reason = " | ".join(why)
    else:
        reason = ""
    return {"admitted": admitted, "score": score, "breakdown": breakdown,
            "critic": cv.verdict, "objections": cv.objections, "reason": reason}
