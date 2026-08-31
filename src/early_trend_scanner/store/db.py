"""SQLite persistence: compact signal records, labels, daily metrics, model blobs.

Writes are rare (signals, labels, checkpoints) and always run off the event
loop via asyncio.to_thread — nothing here sits on the alert path. Retention
limits keep the file bounded.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..engine.state import Signal
from ..ml.labeler import LabelResult

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id      TEXT PRIMARY KEY,
    ts             REAL NOT NULL,
    date_et        TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    direction      INTEGER NOT NULL,
    trigger_verb   TEXT,
    level_kind     INTEGER,
    trigger_price  REAL,
    alert_price    REAL,
    invalidation   REAL,
    vol_ratio      REAL,
    score          REAL,
    prob           REAL,
    model_version  TEXT,
    gate_version   INTEGER,
    suppressed     INTEGER DEFAULT 0,
    features       TEXT,
    resolution     TEXT,
    resolution_ts  REAL,
    resolution_reason TEXT,
    label          INTEGER,
    label_reason   TEXT,
    mfe            REAL,
    mae            REAL,
    lead_time_s    REAL,
    remaining_frac REAL
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date_et);

CREATE TABLE IF NOT EXISTS daily_metrics (
    date_et   TEXT NOT NULL,
    symbol    TEXT NOT NULL,
    payload   TEXT NOT NULL,
    PRIMARY KEY (date_et, symbol)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class SignalStore:
    def __init__(self, path: Path, retention_days: int = 90) -> None:
        self.path = path
        self.retention_days = retention_days
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across asyncio.to_thread workers -> serialize.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._enforce_retention()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def _enforce_retention(self) -> None:
        with self._lock:
            cutoff = time.time() - self.retention_days * 86400
            cur = self._conn.execute("DELETE FROM signals WHERE ts < ?", (cutoff,))
            if cur.rowcount:
                log.info("retention: removed %d old signal rows", cur.rowcount)
            self._conn.execute(
                "DELETE FROM daily_metrics WHERE date_et < date('now', ?)",
                (f"-{self.retention_days} days",),
            )
            self._conn.commit()

    def _write(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ------------------------------------------------------------------ writes

    def record_signal(self, sig: Signal, date_et: str, gate_version: int) -> None:
        self._write(
            """INSERT OR REPLACE INTO signals
               (signal_id, ts, date_et, symbol, direction, trigger_verb, level_kind,
                trigger_price, alert_price, invalidation, vol_ratio, score, prob,
                model_version, gate_version, suppressed, features)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sig.signal_id,
                sig.alert_ts,
                date_et,
                sig.symbol,
                sig.direction,
                sig.trigger_verb,
                sig.level_kind,
                sig.trigger_price,
                sig.alert_price,
                sig.invalidation,
                sig.vol_ratio,
                sig.score,
                sig.prob,
                sig.model_version,
                gate_version,
                int(sig.suppressed),
                json.dumps(sig.features),
            ),
        )

    def record_resolution(self, sig: Signal) -> None:
        self._write(
            """UPDATE signals SET resolution=?, resolution_ts=?, resolution_reason=?
               WHERE signal_id=?""",
            (sig.resolution, sig.resolution_ts, sig.resolution_reason, sig.signal_id),
        )

    def record_label(self, r: LabelResult) -> None:
        self._write(
            """UPDATE signals SET label=?, label_reason=?, mfe=?, mae=?,
               lead_time_s=?, remaining_frac=? WHERE signal_id=?""",
            (
                int(r.label),
                r.reason,
                r.mfe,
                r.mae,
                r.lead_time_s,
                r.remaining_frac,
                r.signal.signal_id,
            ),
        )

    def record_daily_metrics(self, date_et: str, symbol: str, payload: dict[str, Any]) -> None:
        self._write(
            "INSERT OR REPLACE INTO daily_metrics (date_et, symbol, payload) VALUES (?,?,?)",
            (date_et, symbol, json.dumps(payload)),
        )

    def confirmed_before(self, date_et: str, cutoff_ts: float) -> list[tuple]:
        """Delivered CONFIRMED signals alerted before the quiet cutoff (recap)."""
        with self._lock:
            return self._conn.execute(
                "SELECT symbol, direction, ts, alert_price, resolution_ts FROM signals "
                "WHERE date_et=? AND suppressed=0 AND resolution='CONFIRMED' AND ts < ?",
                (date_et, cutoff_ts),
            ).fetchall()

    def set_meta(self, key: str, value: str) -> None:
        self._write("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    # ----------------------------------------------------------------- queries

    def recent_signals(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT signal_id, ts, symbol, direction, trigger_price, resolution,
                          label, lead_time_s, remaining_frac, prob, suppressed
                   FROM signals ORDER BY ts DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        cols = [
            "signal_id",
            "ts",
            "symbol",
            "direction",
            "trigger_price",
            "resolution",
            "label",
            "lead_time_s",
            "remaining_frac",
            "prob",
            "suppressed",
        ]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def label_counts(self) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(label=1),0), COALESCE(SUM(label=0),0) "
                "FROM signals WHERE label IS NOT NULL"
            ).fetchone()
        return int(row[0]), int(row[1])
