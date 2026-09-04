"""L8 PRODUCT - SQLite persistence for rolls, events, and operator decisions.

Small on purpose: three tables, no ORM, no migrations, so a roll can be read
back with the ``sqlite3`` shell during an audit.

The interesting column is ``events.status``. When an operator rejects an event
it becomes a counted false alarm, and that count divided by the inspected
length is the *realised* false-alarm rate - the number the stated budget is
checked against. ``decisions`` is append-only for the same reason: a QA
disposition that changed and left no trace is worse than none at all.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..types import Assertion, Calibration, Event, EventStatus, RollReport, Verdict

__all__ = ["SCHEMA", "Store"]

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS rolls (
    id TEXT PRIMARY KEY,
    sku TEXT NOT NULL,
    started_at TEXT, finished_at TEXT,
    length_m REAL, width_m REAL,
    total_points INTEGER, points_per_100yd2 REAL, verdict TEXT,
    threshold REAL, abstain_low REAL, budget_fa_per_100m REAL,
    metres_per_tile REAL, n_clean_tiles INTEGER, cal_timestamp TEXT,
    gap_warnings INTEGER DEFAULT 0,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_id TEXT NOT NULL REFERENCES rolls(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL,
    along_start_mm REAL, along_end_mm REAL,
    across_start_mm REAL, across_end_mm REAL,
    max_score REAL, confidence REAL,
    assertion TEXT, status TEXT, label TEXT,
    crop_path TEXT, n_frames INTEGER,
    UNIQUE (roll_id, event_id)
);

-- Append-only: a QA disposition that changed and left no trace is worse than
-- no disposition at all.
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_row INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    operator TEXT DEFAULT '',
    at TEXT NOT NULL,
    note TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_roll ON events(roll_id);
"""


class Store:
    """Thin SQLite wrapper. One connection, explicit commits, no magic."""

    def __init__(self, path: Path | str = "linesight.db") -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> Store:
        self.init_schema()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: the API's inspector runs on its own thread
            # and writes events as it finds them.
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def init_schema(self) -> None:
        """Create tables if absent. Idempotent."""
        conn = self._connect()
        conn.executescript(SCHEMA)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- writes ------------------------------------------------------------- #

    def save_roll(self, report: RollReport, config_json: str = "") -> str:
        """Insert or replace a roll and all its events. Returns the roll id."""
        conn = self._connect()
        cal = report.calibration
        conn.execute(
            "INSERT OR REPLACE INTO rolls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                report.roll_id, report.sku, report.started_at, report.finished_at,
                report.roll_length_m, report.width_m, report.total_points,
                report.points_per_100yd2, report.verdict.value,
                cal.threshold if cal else None,
                cal.abstain_low if cal else None,
                cal.budget_fa_per_100m if cal else None,
                cal.metres_per_tile if cal else None,
                cal.n_clean_tiles if cal else None,
                cal.fit_timestamp if cal else None,
                report.gap_warnings,
                config_json or json.dumps(report.meta, default=str),
            ),
        )
        conn.execute("DELETE FROM events WHERE roll_id = ?", (report.roll_id,))
        for event in report.events:
            self._insert_event(conn, report.roll_id, event)
        conn.commit()
        return report.roll_id

    def add_event(self, roll_id: str, event: Event) -> int:
        """Append one event mid-roll. Returns its row id."""
        conn = self._connect()
        row = self._insert_event(conn, roll_id, event)
        conn.commit()
        return row

    @staticmethod
    def _insert_event(conn: sqlite3.Connection, roll_id: str, event: Event) -> int:
        cur = conn.execute(
            "INSERT OR REPLACE INTO events (roll_id, event_id, along_start_mm, along_end_mm,"
            " across_start_mm, across_end_mm, max_score, confidence, assertion, status,"
            " label, crop_path, n_frames) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                roll_id, event.event_id, event.along_start_mm, event.along_end_mm,
                event.across_start_mm, event.across_end_mm, event.max_score,
                event.confidence, event.assertion.value, event.status.value,
                event.label, event.crop_path, event.n_frames,
            ),
        )
        return int(cur.lastrowid)

    def set_event_status(
        self, event_id: int, status: EventStatus, operator: str = "", note: str = ""
    ) -> None:
        """Record an operator decision and append to the audit trail.

        Appends rather than overwrites: a QA disposition that changed and left
        no trace is worse than no disposition at all.
        """
        conn = self._connect()
        conn.execute("UPDATE events SET status = ? WHERE id = ?", (status.value, event_id))
        conn.execute(
            "INSERT INTO decisions (event_row, status, operator, at, note) VALUES (?,?,?,?,?)",
            (
                event_id, status.value, operator,
                datetime.now(timezone.utc).isoformat(timespec="seconds"), note,
            ),
        )
        conn.commit()

    # -- reads -------------------------------------------------------------- #

    def list_rolls(self, limit: int = 50) -> list[dict]:
        """Newest first. Backs ``GET /rolls``."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM rolls ORDER BY COALESCE(finished_at, started_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_roll(self, roll_id: str) -> RollReport | None:
        """Rehydrate a full report, calibration included."""
        conn = self._connect()
        row = conn.execute("SELECT * FROM rolls WHERE id = ?", (roll_id,)).fetchone()
        if row is None:
            return None
        return RollReport(
            roll_id=row["id"], sku=row["sku"],
            roll_length_m=row["length_m"] or 0.0, width_m=row["width_m"] or 0.0,
            events=self.list_events(roll_id),
            total_points=row["total_points"] or 0,
            points_per_100yd2=row["points_per_100yd2"] or 0.0,
            verdict=Verdict(row["verdict"] or "pass"),
            calibration=self.get_calibration(roll_id),
            false_alarms=self.false_alarm_count(roll_id),
            gap_warnings=row["gap_warnings"] or 0,
            started_at=row["started_at"] or "", finished_at=row["finished_at"] or "",
            meta=json.loads(row["config_json"] or "{}"),
        )

    def list_events(self, roll_id: str, status: EventStatus | None = None) -> list[Event]:
        conn = self._connect()
        sql = "SELECT * FROM events WHERE roll_id = ?"
        params: list[object] = [roll_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        rows = conn.execute(sql + " ORDER BY along_start_mm", params).fetchall()
        return [
            Event(
                event_id=r["event_id"],
                along_start_mm=r["along_start_mm"], along_end_mm=r["along_end_mm"],
                across_start_mm=r["across_start_mm"], across_end_mm=r["across_end_mm"],
                max_score=r["max_score"], confidence=r["confidence"],
                assertion=Assertion(r["assertion"]), status=EventStatus(r["status"]),
                label=r["label"] or "unclassified anomaly", crop_path=r["crop_path"],
            )
            for r in rows
        ]

    def get_calibration(self, roll_id: str) -> Calibration | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT threshold, abstain_low, budget_fa_per_100m, metres_per_tile,"
            " n_clean_tiles, sku, cal_timestamp FROM rolls WHERE id = ?", (roll_id,)
        ).fetchone()
        if row is None or row["threshold"] is None:
            return None
        return Calibration(
            threshold=row["threshold"], abstain_low=row["abstain_low"] or 0.0,
            budget_fa_per_100m=row["budget_fa_per_100m"] or 0.0,
            metres_per_tile=row["metres_per_tile"] or 0.0,
            n_clean_tiles=row["n_clean_tiles"] or 0,
            sku=row["sku"] or "", fit_timestamp=row["cal_timestamp"] or "",
        )

    def false_alarm_count(self, roll_id: str) -> int:
        """Operator-rejected events. The live counter's numerator."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE roll_id = ? AND status = ?",
            (roll_id, EventStatus.REJECTED.value),
        ).fetchone()
        return int(row["n"])
