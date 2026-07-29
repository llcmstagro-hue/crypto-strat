"""
Гипотеза = конфиг + СЕТКА параметров, по которой её разрешено подбирать.

Сетка обязана быть частью гипотезы, а не появляться по ходу дела. Причина
прямая: барьер 5 (multiple testing) требует знать, СКОЛЬКО проб было сделано.
Если сетку можно молча расширить после того, как увидел результат, — счётчик
проб теряет смысл, а вместе с ним и вся поправка. Поэтому сетка объявляется
заранее и логируется.

TrialLog хранит счётчик перебранных конфигов на диске: он должен переживать
перезапуски, иначе после рестарта система «забудет», что уже перебрала 400
вариантов, и занизит планку победителю.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field

from ..engine.config import StrategyConfig


@dataclass
class Hypothesis:
    config: StrategyConfig
    grid: dict = field(default_factory=dict)     # "entry.params.period": [10,20,30]
    source: str = ""                             # откуда идея (р.6.5)
    notes: str = ""
    grid_cap: int = 36                           # потолок числа комбинаций

    @property
    def name(self) -> str:
        return self.config.name

    def variants(self) -> list[StrategyConfig]:
        """Все комбинации сетки. Порядок детерминирован — прогон воспроизводим."""
        if not self.grid:
            return [self.config]
        paths = sorted(self.grid)
        combos = list(itertools.product(*(self.grid[p] for p in paths)))
        if len(combos) > self.grid_cap:
            raise ValueError(
                f"{self.name}: сетка даёт {len(combos)} комбинаций > потолка "
                f"{self.grid_cap}. Сузь сетку — широкий перебор поднимает планку "
                f"барьера 5 и обычно означает, что идея не сформулирована."
            )
        out = []
        for combo in combos:
            cfg = self.config
            for path, val in zip(paths, combo):
                cfg = cfg.with_param(path, val)
            cfg.name = f"{self.config.name}|" + ",".join(
                f"{p.split('.')[-1]}={v}" for p, v in zip(paths, combo))
            out.append(cfg)
        return out

    def neighbours(self, cfg: StrategyConfig) -> dict[str, list]:
        """Соседи конфига по сетке (±1 шаг) — вход для барьера стабильности."""
        out = {}
        for path, values in self.grid.items():
            try:
                cur = cfg.get_param(path)
                i = values.index(cur)
            except (ValueError, KeyError):
                continue
            nb = []
            if i > 0:
                nb.append(values[i - 1])
            if i < len(values) - 1:
                nb.append(values[i + 1])
            if nb:
                out[path] = nb
        return out


class TrialLog:
    """Счётчик перебранных конфигов. Без него метрики победителя не значат ничего."""

    def __init__(self, path: str = "./trials.json"):
        self.path = path
        self.data = {"total": 0, "by_tag": {}, "hypotheses": 0}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def record(self, n: int, tag: str = "") -> None:
        self.data["total"] += int(n)
        self.data["by_tag"][tag] = self.data["by_tag"].get(tag, 0) + int(n)
        self._save()

    def record_hypothesis(self) -> None:
        self.data["hypotheses"] = self.data.get("hypotheses", 0) + 1
        self._save()

    def reset(self) -> None:
        self.data = {"total": 0, "by_tag": {}, "hypotheses": 0}
        self._save()

    @property
    def total(self) -> int:
        return int(self.data["total"])

    @property
    def hypotheses(self) -> int:
        return int(self.data.get("hypotheses", 0))

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
