"""
tests/integration/test_api_logs.py

GET /runs/{run_id}/logs and the report's raw-log links (S-6), through FastAPI's TestClient.
Every database and log file is a temp path; no server, no Ollama, no model.

Run (repo root):  python -m pytest tests/integration/test_api_logs.py -v
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from ragregress.api.settings import ApiSettings
from ragregress.core.logging import run_log_dir, run_log_path
from ragregress.main import create_app
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

FAKE_KEY = "sk" + "-" + "A1b2C3d4E5f6G7h8I9j0K1l2"          # assembled at run time (scripts/check_secrets.py)


def noop_executor(config, run_id, on_row_done):
    return None


@pytest.fixture()
def api(tmp_path):
    db = tmp_path / "api.db"
    settings = ApiSettings(db_path=db, configs_dir=tmp_path / "configs", passages_path=tmp_path / "no_passages.json",
                           splits_dir=tmp_path / "no_splits", targets_path=None, top=3)
    app = create_app(settings, executor=noop_executor, configure=False)
    with TestClient(app) as client:
        yield type("Api", (), {"client": client, "repo": Repository(db), "db": db, "log_dir": run_log_dir(db)})


def add_run(repo: Repository, run_id: str, *, recall: float = 1.0) -> None:
    repo.create_run(Run(run_id=run_id, created_at="2026-11-01T00:00:00+00:00", is_dev=False, k=5, max_workers=1,
                        config={"eval_split": "val", "judge_model": "j", "top_k": 5,
                                "pipeline": {"id": "builtin_rag", "params": {"chunk_size": 500}}},
                        config_hash=f"hash-{run_id}", run_status="healthy"))
    scores = {"recall_at_k": recall, "mrr": recall, "token_prf1": 0.5, "faithfulness": 0.9}
    repo.save_query_results([
        QueryResult(run_id=run_id, qa_id="q0", question="question 0?", query_status="judged", failure_mode="pass",
                    category="simple", generation_answer="an answer",
                    metric_results={m: {"value": v} for m, v in scores.items()})])
    repo.save_metrics(run_id, scores)


def write_log(log_dir, run_id: str, records: List[Any]) -> None:
    path = run_log_path(run_id, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in records) + "\n", encoding="utf-8")


def rec(stage: str, query_id: Optional[str], level: str = "INFO", **extra) -> Dict[str, Any]:
    return {"run_id": "r1", "stage": stage, "query_id": query_id, "level": level, "event": f"{stage} {query_id}", **extra}


@pytest.fixture()
def logged(api):
    add_run(api.repo, "r1")
    write_log(api.log_dir, "r1", [rec("run", None), rec("retrieve", "q0"), rec("generate", "q0"),
                                  rec("judge", "q0", "WARNING"), rec("retrieve", "q1"), rec("judge", "q1")])
    return api


# ---------------------------------------------------------------- the endpoint

def test_all_records_are_returned_in_file_order(logged):
    body = logged.client.get("/runs/r1/logs").json()
    assert (body["run_id"], body["total"], body["count"], body["truncated"]) == ("r1", 6, 6, False)
    assert [r["stage"] for r in body["records"]] == ["run", "retrieve", "generate", "judge", "retrieve", "judge"]


def test_filter_by_query_id(logged):
    records = logged.client.get("/runs/r1/logs", params={"query_id": "q0"}).json()["records"]
    assert [r["stage"] for r in records] == ["retrieve", "generate", "judge"]


def test_filters_combine_and_level_is_case_insensitive(logged):
    body = logged.client.get("/runs/r1/logs", params={"query_id": "q0", "stage": "judge", "level": "warning"}).json()
    assert [r["level"] for r in body["records"]] == ["WARNING"] and body["total"] == 1


def test_limit_keeps_the_first_records_and_says_truncated(logged):
    body = logged.client.get("/runs/r1/logs", params={"limit": 2}).json()
    assert (body["total"], body["count"], body["truncated"]) == (6, 2, True)
    assert [r["stage"] for r in body["records"]] == ["run", "retrieve"]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 5001}, {"limit": "x"}])
def test_bad_limit_is_422(logged, params):
    assert logged.client.get("/runs/r1/logs", params=params).status_code == 422


def test_records_are_redacted_again_on_output(api):
    add_run(api.repo, "r1")
    write_log(api.log_dir, "r1", [rec("generate", "q0", error=f"failed with key {FAKE_KEY}")])
    text = api.client.get("/runs/r1/logs").text
    assert FAKE_KEY not in text and "[REDACTED]" in text


def test_a_broken_or_non_object_line_is_kept_as_text(api):
    add_run(api.repo, "r1")
    write_log(api.log_dir, "r1", [rec("run", None), '{"stage": "judge", "query_i', "[1, 2]"])
    records = api.client.get("/runs/r1/logs").json()["records"]
    assert [list(r) for r in records[1:]] == [["raw"], ["raw"]]


def test_unknown_run_is_404(api):
    assert api.client.get("/runs/ghost/logs").status_code == 404


def test_a_log_file_without_a_stored_run_is_not_served(api):
    write_log(api.log_dir, "orphan", [rec("run", None)])
    assert api.client.get("/runs/orphan/logs").status_code == 404


def test_a_stored_run_without_a_log_file_is_404_with_its_own_message(api):
    add_run(api.repo, "old")
    response = api.client.get("/runs/old/logs")
    assert response.status_code == 404 and "log file" in response.json()["detail"]


@pytest.mark.parametrize("bad", [".hidden", "a b", "x" * 129])
def test_a_malformed_run_id_is_422(api, bad):
    assert api.client.get(f"/runs/{bad}/logs").status_code == 422


def test_an_id_with_a_slash_cannot_reach_a_file_outside_the_log_folder(api):
    (api.db.parent / "secret.jsonl").write_text('{"stage": "x"}\n', encoding="utf-8")
    assert api.client.get("/runs/..%2Fsecret/logs").status_code in (404, 422)


# ---------------------------------------------------------------- the report links to it

def test_the_api_report_links_each_drill_down_to_its_log_records(api):
    add_run(api.repo, "base")
    add_run(api.repo, "cand", recall=0.0)
    html = api.client.get("/reports/base__vs__cand").text
    assert '<a href="/runs/cand/logs?query_id=q0">raw log</a>' in html
    assert "available Day 13" not in html