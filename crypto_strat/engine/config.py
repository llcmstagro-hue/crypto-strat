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

# --------------------------------------------------------------------------- #
# ИЗДЕРЖКИ: две модели, обе с полами
# --------------------------------------------------------------------------- #
# Модель "flat" — как было в прогонах 1–10: плоские издержки на обе стороны,
# round-trip >= 30 bps. Оставлена, чтобы старые конфиги из базы знаний
# воспроизводились байт в байт. Новые гипотезы её не используют.
MIN_ROUND_TRIP_BPS = 30.0
LEGACY_COSTS = {
    "model": "flat",
    "fee_bps_per_side": 6.0,
    "slip_bps_per_side": 9.0,
    "funding_default_8h": 0.0001,
}

# Модель "maker_taker" — фактические ставки аккаунта на Bybit (перпы).
# Издержка зависит от того, КАК исполнилась заявка, а не от названия стратегии:
#   * покоящийся лимит исполняется мейкером и БЕЗ проскальзывания — цена
#     заявки известна заранее, хуже неё не дадут;
#   * рыночная и стоп-заявка исполняются тейкером и С проскальзыванием.
#
# ⚠️ ЧТО ЗДЕСЬ УТОЧНЕНО, А ЧТО НЕТ. Уточнена только КОМИССИЯ — это факт о
# тарифе, а не предположение. Проскальзывание оставлено ровно тем же (9 bps),
# каким было в прошлых прогонах. Снижать его «раз уж мы уточняем модель»
# нельзя: комиссия задана биржей, а проскальзывание — это наша оценка
# неизвестного, и подкручивать её в свою пользу — тот же самообман, что
# снижение порогов.
#
# Следствие, о котором стоит помнить при чтении результатов: для РЫНОЧНЫХ
# входов новая модель почти ничего не меняет (14.5 bps на сторону против
# прежних 15.0), поэтому вердикты по трендовым стратегиям остаются как были.
# Выигрывают только те, кто реально входит покоящимся лимитом.
MIN_MAKER_FEE_BPS = 2.0      # 0.02% Bybit maker
MIN_TAKER_FEE_BPS = 5.5      # 0.055% Bybit taker
MIN_SLIP_BPS = 9.0           # проскальзывание тейкерного исполнения. НЕ ПОНИЖАТЬ.

DEFAULT_COSTS = {
    "model": "maker_taker",
    "maker_fee_bps": 2.0,
    "taker_fee_bps": 5.5,
    "slip_bps": 9.0,
    # Поправка на ОЧЕРЕДЬ. Мейкер стоит в стакане, и того, что цена «коснулась»
    # уровня, для исполнения мало: перед нами может стоять чужой объём.
    # Требуем, чтобы цена прошла СКВОЗЬ заявку хотя бы на 1 bp. Это ужесточение,
    # а не послабление: часть касаний перестаёт считаться сделками.
    "maker_through_bps": 1.0,
    "funding_default_8h": 0.0001,
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

        # какая модель издержек: старые конфиги узнаются по своим ключам
        legacy = ("fee_bps_per_side" in self.costs
                  or "slip_bps_per_side" in self.costs
                  or self.costs.get("model") == "flat")
        base = LEGACY_COSTS if legacy else DEFAULT_COSTS
        self.costs = {**base, **self.costs}

        if self.is_flat():
            rt = self.round_trip_bps()
            if rt < MIN_ROUND_TRIP_BPS - 1e-9:
                raise ConfigError(
                    f"round-trip издержки {rt:.1f} bps < минимума "
                    f"{MIN_ROUND_TRIP_BPS} bps. Занижать издержки запрещено "
                    f"(р.4 барьер 7).")
        else:
            for key, floor in (("maker_fee_bps", MIN_MAKER_FEE_BPS),
                               ("taker_fee_bps", MIN_TAKER_FEE_BPS),
                               ("slip_bps", MIN_SLIP_BPS)):
                if float(self.costs[key]) < floor - 1e-9:
                    raise ConfigError(
                        f"{key} = {self.costs[key]} bps < минимума {floor} bps. "
                        f"Занижать издержки запрещено (р.4 барьер 7). Реальный "
                        f"тариф — законное уточнение, оптимистичная оценка — нет.")
        if not 0 < self.sizing["risk_pct"] <= 0.05:
            raise ConfigError("risk_pct вне (0, 0.05]")

    # ------------------------------------------------------------------ #
    def is_flat(self) -> bool:
        return self.costs.get("model", "flat") == "flat"

    def entry_cost_rate(self, kind: str) -> float:
        """Доля нотионала, теряемая на ВХОДЕ. Зависит от типа исполнения.

        Ключевое различие: покоящийся лимит («limit») — мейкер и без
        проскальзывания, потому что цена заявки известна заранее и хуже неё
        исполнения не будет. Рыночная и стоп-заявка («market», «stop») —
        тейкер и с проскальзыванием.
        """
        if self.is_flat():
            return self.cost_rate_per_side()
        if kind == "limit":
            return float(self.costs["maker_fee_bps"]) / 10_000.0
        return (float(self.costs["taker_fee_bps"])
                + float(self.costs["slip_bps"])) / 10_000.0

    def exit_cost_rate(self) -> float:
        """Выход всегда тейкерный: стоп, трейлинг, выход по времени и по
        структуре — всё это исполняется по рынку. Тейк-профит формально мог бы
        стоять лимитом, но записывать его в мейкеры мы НЕ будем: это дало бы
        бесплатную скидку ровно на прибыльных сделках."""
        if self.is_flat():
            return self.cost_rate_per_side()
        return (float(self.costs["taker_fee_bps"])
                + float(self.costs["slip_bps"])) / 10_000.0

    def round_trip_bps(self, entry_kind: str = "market") -> float:
        if self.is_flat():
            return 2.0 * (self.costs["fee_bps_per_side"]
                          + self.costs["slip_bps_per_side"])
        return 10_000.0 * (self.entry_cost_rate(entry_kind) + self.exit_cost_rate())

    def maker_through(self) -> float:
        """Доля цены, на которую рынок обязан пройти СКВОЗЬ лимитную заявку,
        чтобы считать её исполненной. В плоской модели 0 — чтобы прогоны 1–10
        воспроизводились без изменений."""
        if self.is_flat():
            return 0.0
        return float(self.costs.get("maker_through_bps", 1.0)) / 10_000.0

    def cost_rate_per_side(self) -> float:
        if self.is_flat():
            return (self.costs["fee_bps_per_side"]
                    + self.costs["slip_bps_per_side"]) / 10_000.0
        return self.exit_cost_rate()

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
