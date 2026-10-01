"""
SQLite schema and connection helpers

Three tables:
- runs: one row per pipeline run (Pipeline.run() invocation)
- run_metrics: aggregated metric values per run (one row per metric)
- run_query_results: one row per QA pair per run

"""

from __future__ import annotations
 
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Generator

DB_PATH_DEFAULT = Path("ragregress.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    is_dev             INTEGER NOT NULL,
    k                  INTEGER NOT NULL,
    max_workers        INTEGER NOT NULL,
    config_json        TEXT NOT NULL,
    run_status         TEXT,
    run_status_detail  TEXT
);
 
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id       TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    value        REAL,
    PRIMARY KEY (run_id, metric_name),
    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
);
 
CREATE TABLE IF NOT EXISTS run_query_results (
    run_id               TEXT NOT NULL,
    qa_id                TEXT NOT NULL,
    question             TEXT,
    query_status         TEXT,
    failure_mode         TEXT,
    last_error           TEXT,
    retrieved_ids_json   TEXT,
    gold_ids_json        TEXT,
    generation_answer    TEXT,
    metric_results_json  TEXT,
    latency_json         TEXT,
    PRIMARY KEY (run_id, qa_id),
    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
);
"""

_MIGRATIONS = [
    "ALTER TABLE run_query_results ADD COLUMN category TEXT;",
]

def get_connection(db_path: str | Path = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """Open a connection to db

    Parameters:
        db_path: path to database (.db file)
    
    Returns:
        sqlite3.Connection
    """

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def transaction(db_path: str | Path = DB_PATH_DEFAULT) -> Generator[sqlite3.Connection, None, None]:
    # Open a connection for one write, commit on success, roll back on exception, then close.

    with closing(get_connection(db_path)) as conn:
        with conn:
            yield conn

def _run_migrations(conn: sqlite3.Connection) -> None:
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise


def init_db(db_path: str | Path = DB_PATH_DEFAULT) -> None:
    """Create the runs / run_metrics / run_query_results tables if missing.
 
    Idempotent -- safe to call on every Pipeline.__init__.
    """
    with closing(get_connection(db_path)) as conn:
        with conn:
            conn.executescript(_SCHEMA)
            _run_migrations(conn)