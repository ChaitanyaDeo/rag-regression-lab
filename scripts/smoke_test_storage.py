"""
scripts/smoke_test_storage.py

Storage-layer smoke test (Task 10/11). Uses a temporary SQLite file per run
(not :memory:, so foreign-key/cascade behavior is exercised the same way it
will be in production). Run directly:

    python scripts/smoke_test_storage.py

Follows this project's existing convention (smoke_test_pipeline.py,
smoke_test_evaluate.py): plain functions, plain asserts, no pytest.
"""

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from ragregress.storage.database import get_connection, init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository


def _new_repo(tmp_dir: Path) -> tuple[Repository, Path]:
    db_path = tmp_dir / "test.db"
    init_db(db_path)
    return Repository(db_path), db_path


def _sample_run(run_id: str = "run-1") -> Run:
    return Run(
        run_id=run_id,
        created_at="2026-09-25T00:00:00Z",
        is_dev=True,
        k=5,
        max_workers=1,
        config={"is_dev": True, "k": 5, "passages": None},
    )


def check_init_creates_tables(tmp_dir: Path) -> None:
    _, db_path = _new_repo(tmp_dir)
    with closing(get_connection(db_path)) as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "run_metrics", "run_query_results"} <= tables
    print("[ok] init_db creates all 3 tables")


def check_foreign_keys_enabled(tmp_dir: Path) -> None:
    _, db_path = _new_repo(tmp_dir)
    with closing(get_connection(db_path)) as conn:
        (fk_on,) = conn.execute("PRAGMA foreign_keys").fetchone()
    assert fk_on == 1
    print("[ok] foreign_keys pragma is ON")


def check_run_insert_and_get(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    run = _sample_run()
    repo.create_run(run)
    fetched = repo.get_run("run-1")
    assert fetched is not None
    assert fetched.run_id == "run-1"
    assert fetched.config == {"is_dev": True, "k": 5, "passages": None}
    assert fetched.run_status is None
    print("[ok] run insert + get round-trips, including JSON config field")


def check_run_status_update(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.update_run_status("run-1", "healthy", None)
    fetched = repo.get_run("run-1")
    assert fetched.run_status == "healthy"
    assert fetched.run_status_detail is None

    try:
        repo.update_run_status("does-not-exist", "healthy")
        raise AssertionError("expected KeyError for unknown run_id")
    except KeyError:
        pass
    print("[ok] update_run_status updates in place and rejects unknown run_id")


def check_run_upsert_does_not_cascade_wipe(tmp_dir: Path) -> None:
    """create_run on an existing run_id must NOT cascade-delete its
    metrics/query_results -- this is exactly why create_run uses a real
    UPSERT instead of INSERT OR REPLACE."""
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.save_metrics("run-1", {"faithfulness": 1.0})
    repo.create_run(_sample_run())  # re-submit same run_id, e.g. a Studio re-run
    metrics = repo.get_metrics("run-1")
    assert len(metrics) == 1 and metrics[0].value == 1.0
    print("[ok] re-creating an existing run_id does not wipe its metrics")


def check_metrics_insert_and_get(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.save_metrics("run-1", {"faithfulness": 0.667, "recall_at_k": float("nan")})
    metrics = {m.metric_name: m.value for m in repo.get_metrics("run-1")}
    assert metrics["faithfulness"] == 0.667
    assert metrics["recall_at_k"] is None  # NaN stored as NULL
    print("[ok] metrics round-trip; NaN stored as NULL")


def check_query_results_insert_and_get(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    qr = QueryResult(
        run_id="run-1",
        qa_id="qa-1",
        question="q?",
        query_status="judged",
        retrieved_ids=["p1", "p2"],
        gold_ids=["p1"],
        generation_answer="the answer",
        metric_results={"faithfulness": {"value": 1.0, "reason": None}},
        latency={"retrieve": 12.3, "judge": 45.6},
    )
    repo.save_query_result(qr)
    fetched = repo.get_query_results("run-1")
    assert len(fetched) == 1
    got = fetched[0]
    assert got.retrieved_ids == ["p1", "p2"]
    assert got.metric_results == {"faithfulness": {"value": 1.0, "reason": None}}
    assert got.latency == {"retrieve": 12.3, "judge": 45.6}
    print("[ok] query result insert + get round-trips, including JSON fields")


def check_multiple_query_results_same_run(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.save_query_results(
        [QueryResult(run_id="run-1", qa_id=f"qa-{i}") for i in range(3)]
    )
    fetched = repo.get_query_results("run-1")
    assert {r.qa_id for r in fetched} == {"qa-0", "qa-1", "qa-2"}
    print("[ok] multiple query results attach to the same run")


def check_delete_run_cascades(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.save_metrics("run-1", {"faithfulness": 1.0})
    repo.save_query_result(QueryResult(run_id="run-1", qa_id="qa-1"))

    repo.delete_run("run-1")

    assert repo.get_run("run-1") is None
    assert repo.get_metrics("run-1") == []
    assert repo.get_query_results("run-1") == []
    print("[ok] deleting a run cascades to its metrics and query results")


def check_duplicate_pk_and_rollback(tmp_dir: Path) -> None:
    """One good row + one duplicate-PK row in the same batch: the whole
    executemany must roll back, not partially commit."""
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.save_query_result(QueryResult(run_id="run-1", qa_id="qa-1"))

    try:
        repo.save_query_results(
            [
                QueryResult(run_id="run-1", qa_id="qa-2"),  # would succeed alone
                QueryResult(run_id="run-1", qa_id="qa-1"),  # duplicate PK -> fails
            ]
        )
        raise AssertionError("expected sqlite3.IntegrityError for duplicate PK")
    except sqlite3.IntegrityError:
        pass

    fetched = repo.get_query_results("run-1")
    assert [r.qa_id for r in fetched] == ["qa-1"]  # qa-2 was rolled back, not left behind
    print("[ok] duplicate PK raises IntegrityError and rolls back the whole batch")


def check_optional_fields_default_correctly(tmp_dir: Path) -> None:
    repo, _ = _new_repo(tmp_dir)
    repo.create_run(_sample_run())
    repo.save_query_result(QueryResult(run_id="run-1", qa_id="qa-1"))
    got = repo.get_query_results("run-1")[0]
    assert got.question is None
    assert got.failure_mode is None
    assert got.retrieved_ids == []
    assert got.metric_results == {}
    print("[ok] optional/NULL fields default correctly on round-trip")


def check_query_result_fk_violation() -> None:
    """A query result for a run_id that was never created must fail the
    foreign-key constraint, not silently insert an orphan row."""
    with tempfile.TemporaryDirectory() as d:
        repo, _ = _new_repo(Path(d))
        try:
            repo.save_query_result(QueryResult(run_id="no-such-run", qa_id="qa-1"))
            raise AssertionError("expected sqlite3.IntegrityError for missing run_id")
        except sqlite3.IntegrityError:
            pass
    print("[ok] query result for a nonexistent run_id raises a foreign-key IntegrityError")


def main() -> None:
    checks = [
        check_init_creates_tables,
        check_foreign_keys_enabled,
        check_run_insert_and_get,
        check_run_status_update,
        check_run_upsert_does_not_cascade_wipe,
        check_metrics_insert_and_get,
        check_query_results_insert_and_get,
        check_multiple_query_results_same_run,
        check_delete_run_cascades,
        check_duplicate_pk_and_rollback,
        check_optional_fields_default_correctly,
    ]
    for check in checks:
        with tempfile.TemporaryDirectory() as d:
            check(Path(d))
    check_query_result_fk_violation()
    print("\n=== Storage smoke test PASSED ===")


if __name__ == "__main__":
    main()