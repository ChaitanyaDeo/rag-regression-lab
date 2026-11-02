"""
tests/integration/test_repository_list_runs.py

Repository.list_runs(), the read-only list behind GET /runs.
Temporary SQLite only; the real ragregress.db is never touched.

Run (repo root):  python -m pytest tests/integration/test_repository_list_runs.py -v
"""

from __future__ import annotations

import pytest

from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

@pytest.fixture()
def repo(tmp_path) -> Repository:
    db = tmp_path / "list_runs.db"
    init_db(db)
    return Repository(db)


def add(repo: Repository, run_id: str, created_at: str, **kwargs) -> None:
    repo.create_run(Run(run_id=run_id, created_at=created_at, is_dev=kwargs.pop("is_dev", False), k=5,
                        max_workers=1, config=kwargs.pop("config", {"eval_split": "val"}), **kwargs))


def test_an_empty_database_lists_nothing(repo):
    assert repo.list_runs() == []


def test_runs_come_back_newest_first_and_ties_break_on_run_id_descending(repo):
    add(repo, "b", "2026-10-02T00:00:00")
    add(repo, "a", "2026-10-03T00:00:00")
    add(repo, "c", "2026-10-02T00:00:00")
    add(repo, "d", "2026-10-01T00:00:00")
    assert [r.run_id for r in repo.list_runs()] == ["a", "c", "b", "d"]


def test_every_run_field_round_trips_and_dev_runs_are_included(repo):
    add(repo, "r", "2026-10-02T00:00:00", is_dev=True, config={"eval_split": "dev", "pipeline": {"id": "x"}},
        config_hash="h", run_status="healthy", run_status_detail="fine")
    (run,) = repo.list_runs()
    assert run == repo.get_run("r")
    assert (run.is_dev, run.config_hash, run.run_status, run.run_status_detail) == (True, "h", "healthy", "fine")
    assert run.config == {"eval_split": "dev", "pipeline": {"id": "x"}}


def test_listing_does_not_change_anything_and_ignores_query_results(repo):
    add(repo, "r", "2026-10-02T00:00:00")
    repo.save_query_results([QueryResult(run_id="r", qa_id="q1")])
    assert len(repo.list_runs()) == 1 and len(repo.list_runs()) == 1
    assert len(repo.get_query_results("r")) == 1