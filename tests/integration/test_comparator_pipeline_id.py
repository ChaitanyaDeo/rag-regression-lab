"""
tests/integration/test_comparator_pipeline_id.py

compare warns when the two runs come from different pipelines (post-Day 14, Q4).
The warning is informational: nothing is ungated and the metrics are still compared.

Real (temporary) SQLite database through the project's Repository; the real ragregress.db is never touched.

Run (repo root):  python -m pytest tests/integration/test_comparator_pipeline_id.py -v
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from ragregress.evaluation.comparator import compare_runs
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository


@pytest.fixture()
def repo(tmp_path) -> Repository:
    db = tmp_path / "test_comparator_pipeline_id.db"
    init_db(db)
    return Repository(db)


def config(pipeline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"eval_split": "val", "judge_model": "qwen2.5:7b", "top_k": 5}
    if pipeline is not None:
        cfg["pipeline"] = pipeline
    return cfg


def add_run(repo: Repository, run_id: str, pipeline: Optional[Dict[str, Any]], recall: float) -> None:
    repo.create_run(Run(run_id=run_id, created_at="2026-11-03T00:00:00", is_dev=False, k=5,
                        max_workers=1, config=config(pipeline), config_hash="hash"))
    repo.save_query_results([QueryResult(
        run_id=run_id, qa_id="q1", question="q?", query_status="judged", category="cat",
        generation_answer="an answer", metric_results={"recall_at_k": {"value": recall, "reason": ""}},
    )])


def pipeline_warnings(result):
    return [w for w in result.warnings if "pipeline.id differs" in w]


def test_different_pipeline_ids_warn(repo):
    add_run(repo, "base", {"id": "builtin_rag", "params": {}}, recall=1.0)
    add_run(repo, "cand", {"id": "my_rag", "params": {}}, recall=0.0)
    result = compare_runs(repo, "base", "cand")

    (warning,) = pipeline_warnings(result)
    assert warning == ("pipeline.id differs (baseline='builtin_rag', candidate='my_rag'); "
                       "the verdict compares two pipelines, not a config change")
    assert ("pipeline.id", "builtin_rag", "my_rag") in [(c.key, c.before, c.after) for c in result.config_changes]


def test_different_pipeline_ids_ungate_nothing_and_still_compare(repo):
    add_run(repo, "base", {"id": "builtin_rag", "params": {}}, recall=1.0)
    add_run(repo, "cand", {"id": "my_rag", "params": {}}, recall=0.0)
    result = compare_runs(repo, "base", "cand")

    assert result.ungated == {}
    (recall,) = [m for m in result.metrics if m.name == "recall_at_k"]
    assert recall.delta == pytest.approx(-1.0)


def test_same_pipeline_id_does_not_warn(repo):
    add_run(repo, "base", {"id": "builtin_rag", "params": {"chunk_size": 500}}, recall=1.0)
    add_run(repo, "cand", {"id": "builtin_rag", "params": {"chunk_size": 1000}}, recall=1.0)
    result = compare_runs(repo, "base", "cand")
    assert pipeline_warnings(result) == []
    assert result.warnings == []


@pytest.mark.parametrize("base_pipeline, cand_pipeline", [
    (None, {"id": "my_rag", "params": {}}),                         # legacy baseline: no pipeline at all
    ({"name": "baseline", "params": {}}, {"id": "my_rag", "params": {}}),   # old shape without `id`
    ({"id": "builtin_rag", "params": {}}, {"name": "baseline", "params": {}}),
])
def test_missing_pipeline_id_on_either_side_does_not_warn(repo, base_pipeline, cand_pipeline):
    add_run(repo, "base", base_pipeline, recall=1.0)
    add_run(repo, "cand", cand_pipeline, recall=1.0)
    assert pipeline_warnings(compare_runs(repo, "base", "cand")) == []