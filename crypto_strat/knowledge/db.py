"""
База знаний по гипотезам (р.6.7) — persistent, SQLite.

Зачем она существует (три причины, все практические):
  1. не тестировать один и тот же мусор дважды — дедуп по отпечатку ЛОГИКИ,
     а не по имени: переименованная гипотеза остаётся тем же кандидатом;
  2. со временем видно, какие ТИПЫ идей чаще выживают → куда направлять поиск;
  3. это и есть интеллектуальный актив лаборатории.

Сохраняется КАЖДАЯ гипотеза, включая отсеянные Research Score и Critic'ом
ещё до бэктеста, — отрицательный результат тоже знание, и он тоже стоит
вычислений, если его забыть и повторить.

Отдельно хранится счётчик проб на момент теста (`trials_at_test`): без него
нельзя честно перечитать результат задним числом, потому что планка барьера 5
зависит от того, сколько было перебрано К ТОМУ МОМЕНТУ.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint       TEXT NOT NULL,
    name              TEXT NOT NULL,
    source            TEXT,
    strategy_type     TEXT,
    generation        TEXT,
    tf                TEXT,
    config_json       TEXT,
    grid_json         TEXT,
    rationale         TEXT,
    research_score    INTEGER,
    score_breakdown   TEXT,
    critic_verdict    TEXT,
    critic_objections TEXT,
    tested            INTEGER DEFAULT 0,
    survived          INTEGER DEFAULT 0,
    barriers_passed   INTEGER,
    failed_barriers   TEXT,
    robustness        REAL,
    metrics           TEXT,
    regimes           TEXT,
    reason            TEXT,
    data_span         TEXT,
    trials_at_test    INTEGER,
    created_at        TEXT,
    UNIQUE(fingerprint, tf)
);
CREATE INDEX IF NOT EXISTS idx_survived ON hypotheses(survived);
CREATE INDEX IF NOT EXISTS idx_type     ON hypotheses(strategy_type);
CREATE INDEX IF NOT EXISTS idx_gen      ON hypotheses(generation);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Record:
    fingerprint: str
    name: str
    tf: str
    source: str = ""
    strategy_type: str = ""
    generation: str = ""
    config: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)
    rationale: str = ""
    research_score: int = 0
    score_breakdown: dict = field(default_factory=dict)
    critic_verdict: str = ""
    critic_objections: list = field(default_factory=list)
    tested: bool = False
    survived: bool = False
    barriers_passed: int | None = None
    failed_barriers: list = field(default_factory=list)
    robustness: float | None = None
    metrics: dict = field(default_factory=dict)
    regimes: dict = field(default_factory=dict)
    reason: str = ""
    data_span: str = ""
    trials_at_test: int | None = None


class KnowledgeBase:
    def __init__(self, path: str = "./knowledge.db"):
        self.path = path
        new = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        if new:
            print(f"[база знаний] создана: {os.path.abspath(path)}")

    # ------------------------------------------------------------------ #
    def seen(self, fingerprint: str, tf: str) -> dict | None:
        """Уже тестировали эту ЛОГИКУ на этом ТФ? Возвращает запись или None."""
        cur = self.conn.execute(
            "SELECT * FROM hypotheses WHERE fingerprint=? AND tf=?", (fingerprint, tf))
        row = cur.fetchone()
        return dict(row) if row else None

    def save(self, rec: Record) -> int:
        """Пишет/обновляет запись. Возвращает id.

        Дубликат НЕ затирает оригинал. Отпечаток логики у них общий, а ключ в
        таблице — (fingerprint, tf), поэтому наивный upsert подменял бы
        ПРОТЕСТИРОВАННУЮ запись НЕпротестированной: имя, метрики и вердикт
        оригинала исчезали, и в базе оставался «дубликат самого себя».
        Именно так однажды пропал базовый order_block, оставив в базе только
        своих потомков по Evolution.
        """
        prev = self.seen(rec.fingerprint, rec.tf)
        if prev and prev.get("tested") and not rec.tested:
            # оригинал уже прогнан — новый экземпляр просто отмечаем дубликатом
            objs = sorted(set(json.loads(prev.get("critic_objections") or "[]")
                              + [f"дубликат отклонён: {rec.name}"]))
            self.conn.execute(
                "UPDATE hypotheses SET critic_objections=? WHERE id=?",
                (json.dumps(objs, ensure_ascii=False), prev["id"]))
            self.conn.commit()
            return int(prev["id"])

        payload = (
            rec.fingerprint, rec.name, rec.source, rec.strategy_type, rec.generation,
            rec.tf, json.dumps(rec.config, ensure_ascii=False),
            json.dumps(rec.grid, ensure_ascii=False), rec.rationale,
            int(rec.research_score), json.dumps(rec.score_breakdown, ensure_ascii=False),
            rec.critic_verdict, json.dumps(rec.critic_objections, ensure_ascii=False),
            int(rec.tested), int(rec.survived), rec.barriers_passed,
            json.dumps(rec.failed_barriers, ensure_ascii=False), rec.robustness,
            json.dumps(rec.metrics, ensure_ascii=False, default=str),
            json.dumps(rec.regimes, ensure_ascii=False, default=str),
            rec.reason, rec.data_span, rec.trials_at_test, _now(),
        )
        cols = ("fingerprint,name,source,strategy_type,generation,tf,config_json,"
                "grid_json,rationale,research_score,score_breakdown,critic_verdict,"
                "critic_objections,tested,survived,barriers_passed,failed_barriers,"
                "robustness,metrics,regimes,reason,data_span,trials_at_test,created_at")
        names = cols.split(",")
        # число плейсхолдеров считается из списка колонок, а не пишется руками:
        # рассинхрон «24 колонки / 23 значения» ловится только в рантайме
        assert len(names) == len(payload), (len(names), len(payload))
        self.conn.execute(
            f"INSERT INTO hypotheses ({cols}) VALUES ({','.join('?' * len(names))}) "
            f"ON CONFLICT(fingerprint, tf) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in names if c != "fingerprint"),
            payload)
        self.conn.commit()
        cur = self.conn.execute(
            "SELECT id FROM hypotheses WHERE fingerprint=? AND tf=?", (rec.fingerprint, rec.tf))
        return int(cur.fetchone()["id"])

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        c = self.conn
        q = lambda s, *a: c.execute(s, a).fetchone()[0]
        return {
            "всего гипотез": q("SELECT COUNT(*) FROM hypotheses"),
            "дошли до бэктеста": q("SELECT COUNT(*) FROM hypotheses WHERE tested=1"),
            "отсеяны до бэктеста": q("SELECT COUNT(*) FROM hypotheses WHERE tested=0"),
            "выжили": q("SELECT COUNT(*) FROM hypotheses WHERE survived=1"),
        }

    def by_type(self) -> list[dict]:
        """Какие ТИПЫ идей чаще доживают — это и есть накопленное знание."""
        rows = self.conn.execute("""
            SELECT strategy_type AS тип,
                   COUNT(*) AS всего,
                   SUM(tested) AS тестировано,
                   SUM(survived) AS выжило,
                   ROUND(AVG(CASE WHEN tested=1 THEN barriers_passed END), 2) AS ср_барьеров,
                   ROUND(AVG(CASE WHEN tested=1 THEN robustness END), 3) AS ср_робастность
            FROM hypotheses GROUP BY strategy_type ORDER BY выжило DESC, ср_барьеров DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def failure_reasons(self, limit: int = 15) -> list[tuple]:
        """На каких барьерах чаще всего умирают — куда смотреть в следующий раз."""
        from collections import Counter
        cnt = Counter()
        for r in self.conn.execute(
                "SELECT failed_barriers FROM hypotheses WHERE tested=1"):
            for b in json.loads(r["failed_barriers"] or "[]"):
                cnt[b] += 1
        return cnt.most_common(limit)

    def survivors(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM hypotheses WHERE survived=1 ORDER BY robustness DESC").fetchall()
        return [dict(r) for r in rows]

    def all_tested(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM hypotheses WHERE tested=1 "
            "ORDER BY barriers_passed DESC, robustness DESC").fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
