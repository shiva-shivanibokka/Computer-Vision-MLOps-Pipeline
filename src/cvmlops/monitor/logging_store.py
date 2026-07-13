"""SQLite prediction log — the single source of truth for monitoring & drift."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from cvmlops.config import REPO_ROOT, get_settings
from cvmlops.monitor.features import FEATURE_COLS

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    request_id    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    {", ".join(f"{c} REAL" for c in FEATURE_COLS)}
);
"""


def _db_path() -> Path:
    p = Path(get_settings().prediction_db)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    # WAL + busy_timeout so concurrent prediction writes from the serving endpoint
    # don't raise "database is locked".
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(_SCHEMA)
    return conn


def log_prediction(request_id: str, model_version: str, features: dict[str, float]) -> None:
    cols = ["ts", "request_id", "model_version", *FEATURE_COLS]
    vals = [
        pd.Timestamp.utcnow().isoformat(),
        request_id,
        model_version,
        *[features.get(c) for c in FEATURE_COLS],
    ]
    placeholders = ", ".join("?" * len(cols))
    with closing(_connect()) as conn:
        conn.execute(f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({placeholders})", vals)
        conn.commit()


def load_predictions(limit: int | None = None) -> pd.DataFrame:
    q = "SELECT * FROM predictions ORDER BY id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with closing(_connect()) as conn:
        return pd.read_sql_query(q, conn)


def count() -> int:
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
