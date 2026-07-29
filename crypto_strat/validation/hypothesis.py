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
    # Межрыночным гипотезам нужен свой набор инструментов: ведущий (BTC) не
    # может быть одновременно ведомым — «импульс BTC как сигнал на BTC» это
    # просто моментум, а не межрыночный эффект.
    primary: str | None = None                   # где подбирать параметры
    exclude_symbols: tuple = ()                  # кого убрать из корзины

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
    """Счётчик перебранных конфигов. Без него метрики победителя не значат ничего.

    Считается ДВА числа, и разница между ними существенна:

      * `total`     — сколько конфигов вообще прогнано (нагрузка, для отчёта);
      * `selection` — сколько конфигов РЕАЛЬНО СОРЕВНОВАЛИСЬ за право быть
                      победителем, чей результат мы потом публикуем.

    Поправка на multiple testing (барьер 5) отвечает на вопрос «мы выбрали
    лучшего из N — насколько он мог оказаться лучшим случайно». Значит в N
    входят только те конфиги, которые МОГЛИ стать этим победителем: перебор
    на train внутри гипотезы и все гипотезы между собой.

    НЕ входят внутренние переборы walk-forward: конфиг, выбранный в окне 7,
    меряется на собственном OOS окна 7 и агрегируется — он никогда не
    становится «тем самым победителем», чью метрику мы публикуем. Считать их
    означало бы раздуть N в ~30 раз без статистического основания и сделать
    барьер непроходимым по бухгалтерской причине, а не по существу.

    Порог DSR при этом НЕ меняется (0.95) — правится только смысл N.
    """

    def __init__(self, path: str = "./trials.json"):
        self.path = path
        self.data = {"total": 0, "selection": 0, "by_tag": {}, "hypotheses": 0}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = {**self.data, **json.load(f)}
            except Exception:
                pass

    def record(self, n: int, tag: str = "", selection: bool = True) -> None:
        """selection=False — прогон-диагностика (walk-forward, соседи по сетке):
        считается в нагрузку, но не в пул отбора."""
        n = int(n)
        self.data["total"] += n
        if selection:
            self.data["selection"] = self.data.get("selection", 0) + n
        self.data["by_tag"][tag] = self.data["by_tag"].get(tag, 0) + n
        self._save()

    def record_hypothesis(self) -> None:
        self.data["hypotheses"] = self.data.get("hypotheses", 0) + 1
        self._save()

    def reset(self) -> None:
        self.data = {"total": 0, "selection": 0, "by_tag": {}, "hypotheses": 0}
        self._save()

    @property
    def total(self) -> int:
        return int(self.data["total"])

    @property
    def selection(self) -> int:
        """Пул отбора — вход барьера 5."""
        return int(self.data.get("selection", self.data["total"]))

    @property
    def hypotheses(self) -> int:
        return int(self.data.get("hypotheses", 0))

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
