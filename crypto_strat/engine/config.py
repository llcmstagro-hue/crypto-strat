"""
Формат гипотезы (р.6). Стратегия = КОНФИГ ИЗ БЛОКОВ, не свободный код.

Смысл ограничения: и генератор комбинаций, и веб-поисковик, и рука человека
складывают стратегию из одного и того же набора кубиков. Поэтому любая идея —
из статьи SSRN, из freqtrade или своя — проходит через один движок и один
фильтр, без поблажек по происхождению (р.6.5).

    {
      "name": "donchian_20_htf",
      "direction": "both",                  # long | short | both
      "timeframe": "1h",
      "entry":  {"type": "donchian_breakout", "params": {"period": 20}},
      "stop":   {"type": "atr", "params": {"period": 14, "mult": 2.0}},
      "exit":   {"type": "fixed_rr", "params": {"rr": 2.0, "max_bars": 120}},
      "filters": [{"type": "htf_trend", "params": {"factor": 4, "ema_period": 50}}],
      "sizing": {"risk_pct": 0.01},
      "costs":  {"fee_bps_per_side": 6.0, "slip_bps_per_side": 9.0,
                 "funding_default_8h": 0.0001},
      "meta":   {"source": "...", "notes": "..."}
    }

Издержки: round-trip = 2*(fee+slip) в bps. Минимум 30 bps (0.3% нотионала)
жёстко ЗАШИТ в валидацию — снизить нельзя, конфиг с меньшими издержками
не создастся (р.4 барьер 7, р.7 guardrails).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict

# Минимальный round-trip в базисных пунктах нотионала. НЕ ПОНИЖАТЬ.
MIN_ROUND_TRIP_BPS = 30.0

DEFAULT_COSTS = {
    "fee_bps_per_side": 6.0,        # комиссия тейкера
    "slip_bps_per_side": 9.0,       # проскальзывание
    "funding_default_8h": 0.0001,   # 0.01% / 8ч, если нет исторического funding
}

DEFAULT_SIZING = {"risk_pct": 0.01, "max_leverage": 100.0}


class ConfigError(ValueError):
    pass


@dataclass
class StrategyConfig:
    name: str
    entry: dict
    stop: dict
    exit: dict
    direction: str = "both"
    timeframe: str = "1h"
    filters: list = field(default_factory=list)
    sizing: dict = field(default_factory=lambda: dict(DEFAULT_SIZING))
    costs: dict = field(default_factory=lambda: dict(DEFAULT_COSTS))
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        if self.direction not in ("long", "short", "both"):
            raise ConfigError(f"direction: {self.direction}")
        for k in ("entry", "stop", "exit"):
            blk = getattr(self, k)
            if not isinstance(blk, dict) or "type" not in blk:
                raise ConfigError(f"блок '{k}' должен быть dict с ключом 'type'")
            blk.setdefault("params", {})
        for f in self.filters:
            if not isinstance(f, dict) or "type" not in f:
                raise ConfigError("фильтр должен быть dict с ключом 'type'")
            f.setdefault("params", {})

        self.sizing = {**DEFAULT_SIZING, **self.sizing}
        self.costs = {**DEFAULT_COSTS, **self.costs}

        rt = self.round_trip_bps()
        if rt < MIN_ROUND_TRIP_BPS - 1e-9:
            raise ConfigError(
                f"round-trip издержки {rt:.1f} bps < минимума {MIN_ROUND_TRIP_BPS} bps. "
                "Занижать издержки запрещено (р.4 барьер 7)."
            )
        if not 0 < self.sizing["risk_pct"] <= 0.05:
            raise ConfigError("risk_pct вне (0, 0.05]")

    # ------------------------------------------------------------------ #
    def round_trip_bps(self) -> float:
        return 2.0 * (self.costs["fee_bps_per_side"] + self.costs["slip_bps_per_side"])

    def cost_rate_per_side(self) -> float:
        return (self.costs["fee_bps_per_side"] + self.costs["slip_bps_per_side"]) / 10_000.0

    # ---- работа с параметрами по точечному пути (нужна барьеру стабильности) ---- #
    @staticmethod
    def _step(node, part: str):
        """Шаг по пути. filters — список, поэтому 'filters.2.params.x' требует
        превращения '2' в индекс, иначе путь до параметров фильтров недоступен."""
        if isinstance(node, list):
            return node[int(part)]
        return node[part]

    def get_param(self, path: str):
        node = asdict(self)
        for part in path.split("."):
            node = self._step(node, part)
        return node

    def with_param(self, path: str, value) -> "StrategyConfig":
        d = self.to_dict()
        node = d
        parts = path.split(".")
        for part in parts[:-1]:
            node = self._step(node, part)
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
        d["name"] = f"{self.name}[{path}={value}]"
        return StrategyConfig.from_dict(d)

    def numeric_param_paths(self) -> list[str]:
        """Все числовые параметры блоков — пространство, по которому барьер
        стабильности ищет плато. Издержки и сайзинг сюда не входят: их
        не «подбирают», они заданы реальностью."""
        out = []

        def walk(prefix: str, d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(f"{prefix}.{k}", v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(f"{prefix}.{k}")

        walk("entry", self.entry)
        walk("stop", self.stop)
        walk("exit", self.exit)
        for i, f in enumerate(self.filters):
            walk(f"filters.{i}", f)
        return [p for p in out if ".params." in p]

    def n_rules(self) -> int:
        """Число «правил» = вход + стоп + выход + фильтры.
        Красный флаг подгонки (р.4): >5-6 правил на небольшой выборке."""
        return 3 + len(self.filters)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "StrategyConfig":
        d = copy.deepcopy(d)
        known = {"name", "entry", "stop", "exit", "direction", "timeframe",
                 "filters", "sizing", "costs", "meta"}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"неизвестные ключи конфига: {sorted(unknown)}")
        return StrategyConfig(**d)

    @staticmethod
    def from_json(s: str) -> "StrategyConfig":
        return StrategyConfig.from_dict(json.loads(s))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def fingerprint(self) -> str:
        """Стабильный отпечаток логики без имени — чтобы ловить дубликаты
        гипотез (р.6.5: «не дубликат ли уже стоящего в очереди»)."""
        d = self.to_dict()
        d.pop("name", None)
        d.pop("meta", None)
        return json.dumps(d, sort_keys=True, ensure_ascii=False)
