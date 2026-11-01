from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from ragregress.storage.database import DB_PATH_DEFAULT, get_connection, transaction
from ragregress.storage.models import QueryResult, Run, RunMetric

_INSERT_QUERY_RESULT_SQL = """
    INSERT INTO run_query_results (
        run_id, qa_id, question, query_status, failure_mode, category,
        last_error, retrieved_ids_json, gold_ids_json,
        generation_answer, metric_results_json, latency_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# _UPSERT_RUN_SQL = """
#     INSERT INTO runs (run_id, created_at, is_dev, k, max_workers,
#                        config_json, run_status, run_status_detail)
#     VALUES (:run_id, :created_at, :is_dev, :k, :max_workers,
#             :config_json, :run_status, :run_status_detail)
#     ON CONFLICT(run_id) DO UPDATE SET
#         created_at=excluded.created_at,
#         is_dev=excluded.is_dev,
#         k=excluded.k,
#         max_workers=excluded.max_workers,
#         config_json=excluded.config_json,
#         run_status=excluded.run_status,
#         run_status_detail=excluded.run_status_detail
# """

_UPSERT_RUN_SQL = """
    INSERT INTO runs (run_id, created_at, is_dev, k, max_workers,
                       config_json, config_hash, run_status, run_status_detail)
    VALUES (:run_id, :created_at, :is_dev, :k, :max_workers,
            :config_json, :config_hash, :run_status, :run_status_detail)
    ON CONFLICT(run_id) DO UPDATE SET
        created_at=excluded.created_at,
        is_dev=excluded.is_dev,
        k=excluded.k,
        max_workers=excluded.max_workers,
        config_json=excluded.config_json,
        config_hash=excluded.config_hash,
        run_status=excluded.run_status,
        run_status_detail=excluded.run_status_detail
"""

_UPSERT_METRIC_SQL = """
    INSERT INTO run_metrics (run_id, metric_name, value)
    VALUES (?, ?, ?)
    ON CONFLICT(run_id, metric_name) DO UPDATE SET value = excluded.value
"""

def _dumps(obj: Any) -> str:
    """JSON-encode a value that may contain objects with a __json__ method
    (e.g. MetricResult, evaluation/metrics/base.py), falling back to str()
    for anything else that isn't natively serializable.
    """
    return json.dumps(obj, default=lambda o: o.__json__() if hasattr(o, "__json__") else str(o))

def _query_result_params(r: QueryResult) -> tuple:
    return (
        r.run_id,
        r.qa_id,
        r.question,
        r.query_status,
        r.failure_mode,
        r.category,
        r.last_error,
        _dumps(r.retrieved_ids),
        _dumps(r.gold_ids),
        r.generation_answer,
        _dumps(r.metric_results),
        _dumps(r.latency),
    )

def _row_to_query_result(row: sqlite3.Row) -> QueryResult:
    return QueryResult(
        run_id=row["run_id"],
        qa_id=row["qa_id"],
        question=row["question"],
        query_status=row["query_status"],
        failure_mode=row["failure_mode"],
        last_error=row["last_error"],
        retrieved_ids=json.loads(row["retrieved_ids_json"]) if row["retrieved_ids_json"] else [],
        gold_ids=json.loads(row["gold_ids_json"]) if row["gold_ids_json"] else [],
        generation_answer=row["generation_answer"],
        metric_results=json.loads(row["metric_results_json"]) if row["metric_results_json"] else {},
        latency=json.loads(row["latency_json"]) if row["latency_json"] else {},
        category=row["category"]
    )

def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        run_id = row['run_id'],
        created_at=row["created_at"],
        is_dev=bool(row["is_dev"]),
        k=row["k"],
        max_workers=row["max_workers"],
        config=json.loads(row["config_json"]),
        config_hash=row["config_hash"],
        run_status=row["run_status"],
        run_status_detail=row["run_status_detail"],
    )


class Repository:

    def __init__(self, db_path: str | Path = DB_PATH_DEFAULT):
        self._db_path = db_path


    def create_run(self, run : Run)->None:
        # Insert a run, or update it in place if run_id already exists.

        with transaction(self._db_path) as conn:
            conn.execute(
                _UPSERT_RUN_SQL,
                {
                    "run_id": run.run_id,
                    "created_at": run.created_at,
                    "is_dev": int(run.is_dev),
                    "k": run.k,
                    "max_workers": run.max_workers,
                    "config_json": _dumps(run.config),
                    "config_hash": run.config_hash,
                    "run_status": run.run_status,
                    "run_status_detail": run.run_status_detail,
                }
            )


    def update_run_status(self, run_id:str, run_status:str | None, run_status_detail: str | None = None) -> None:
        with transaction(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE runs SET run_status = ?, run_status_detail = ? WHERE run_id = ?",
                (run_status, run_status_detail, run_id)
            )

            if cur.rowcount == 0:
                raise KeyError(f"No run with run_id={run_id!r}")

    def get_run(self, run_id: str) -> Run | None:
        with closing(get_connection(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None

        return _row_to_run(row)

    def list_runs(self) -> list[Run]:
        """Every stored run, newest first (created_at, then run_id). Read-only: no metrics, no query results.

        Dev runs are included; the caller decides what to show (Run.is_dev).
        """
        with closing(get_connection(self._db_path)) as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC, run_id DESC").fetchall()
        return [_row_to_run(row) for row in rows]

    def delete_run(self, run_id: str) -> None:
        """Delete the run. ON DELETE CASCADE removes its run_metrics and
        run_query_results rows too -- no separate cleanup needed here."""
        with transaction(self._db_path) as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


    # Metrics -----------------------------------------------------------

    def save_metrics(self, run_id:str, metrics: dict[str, float])-> None:
        # metrics: {metric_name: value}. NaN is stored as SQL NULL so aggregate queries (AVG, WHERE value IS NOT NULL) work normally;

        rows = [(run_id, name, None if value != value else value) for name, value in metrics.items()]
        with transaction(self._db_path) as conn:
            conn.executemany(_UPSERT_METRIC_SQL, rows)

    def get_metrics(self, run_id:str) -> list[RunMetric]:
        with closing(get_connection(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT run_id, metric_name, value FROM run_metrics WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return [RunMetric(run_id=r["run_id"], metric_name=r["metric_name"], value=r["value"]) for r in rows]

    # Query Result---------------------------------------------------------

    def save_query_result(self, result: QueryResult) -> None:
        # Insert one query result row
        with transaction(self._db_path) as conn:
            conn.execute(_INSERT_QUERY_RESULT_SQL, _query_result_params(result))


    def save_query_results(self, results: Iterable[QueryResult]) -> None:
        # Save several query results in one transaction -- matches RunState.per_query_results
        with transaction(self._db_path) as conn:
            conn.executemany(_INSERT_QUERY_RESULT_SQL, [_query_result_params(r) for r in results])

    def get_query_results(self, run_id: str) -> list[QueryResult]:
        with closing(get_connection(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM run_query_results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [_row_to_query_result(row) for row in rows]