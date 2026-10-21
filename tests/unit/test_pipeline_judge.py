"""
tests/unit/test_pipeline_judge.py

  A judge crash on ONE metric must not discard the other metrics (D8-F2, option ii).
  every metric crashes -> row stays FAILED / failed (nothing was scored)
  judged row missing a configured metric -> failure_mode `judge_failed`, overriding every other tag
  the crashed metric is absent from metric_results; its error text goes to last_error

Calls Pipeline._judge / _aggregate / _persist directly with stub metrics (no LLM, no graph run).

Run (repo root):  python -m pytest tests/unit/test_pipeline_judge.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import pytest

from ragregress.evaluation.comparator import RowKind, load_snapshot
from ragregress.evaluation.regression import run_validity
from ragregress.pipeline.pipeline import Pipeline
from ragregress.pipeline.pipeline_constants import FailureMode, QueryStatus
from ragregress.storage.models import Run
from ragregress.storage.repositories import Repository


# ---------------------------------------------------------------- stubs

@dataclass
class _Result:
    value: float
    reason: str = ""

    def __json__(self):                      # the repository serialises results through __json__ (see _dumps)
        return {"value": self.value, "reason": self.reason}


class _Metric:
    """Stub with the real metrics' names so the failure-mode chain in _aggregate applies unchanged."""
    required_inputs = {"retrieved_ids"}

    def __init__(self, name: str, value: float = 1.0, boom: bool = False):
        self.name, self._value, self._boom = name, value, boom

    def score(self, **kwargs) -> _Result:
        if self._boom:
            raise RuntimeError(f"{self.name} judge exploded")
        return _Result(self._value)


class _Rag:
    def retrieve(self, question, k=5):
        return []


def make_pipeline(tmp_path, *metrics) -> Pipeline:
    return Pipeline(rag_system=_Rag(), metrics=list(metrics), k=5, db_path=tmp_path / "judge.db")


def query_state(qa_id: str = "q1") -> Dict:
    return {
        "qa_id": qa_id, "question": "q?", "gold_ids": ["p1"], "gold_answers": ["a"], "category": "cat",
        "chunks": [SimpleNamespace(source_passage_id="p1")], "generation": SimpleNamespace(answer="an answer"),
        "latency_ms": {}, "metric_results": {}, "retry_count": {}, "last_error": None,
    }


def judge(pipe: Pipeline, qa_id: str = "q1") -> Dict:
    """Run _judge and return the finished row exactly as the reducer would receive it."""
    return pipe._judge(query_state(qa_id))["per_query_results"][0]


def aggregate(pipe: Pipeline, *rows: Dict) -> Dict:
    return pipe._aggregate({"per_query_results": list(rows)})["aggregated_metrics"]


R, M, T, F = "recall_at_k", "mrr", "token_prf1", "faithfulness"


# ---------------------------------------------------------------- _judge

def test_all_metrics_ok_is_unchanged(tmp_path):
    pipe = make_pipeline(tmp_path, _Metric(R), _Metric(F))
    row = judge(pipe)

    assert row["query_status"] == QueryStatus.JUDGED
    assert set(row["metric_results"]) == {R, F}
    assert not row.get("last_error")
    aggregate(pipe, row)
    assert row["failure_mode"] == FailureMode.PASS


def test_one_crashed_metric_keeps_the_others(tmp_path):
    pipe = make_pipeline(tmp_path, _Metric(R), _Metric(T, 0.8), _Metric(F, boom=True))
    row = judge(pipe)

    assert row["query_status"] == QueryStatus.JUDGED
    assert set(row["metric_results"]) == {R, T}                      # faithfulness absent, not None
    assert row["last_error"].startswith("faithfulness: RuntimeError(")
    assert "faithfulness judge exploded" in row["last_error"]


def test_a_crash_does_not_stop_the_metrics_after_it(tmp_path):
    pipe = make_pipeline(tmp_path, _Metric(R, boom=True), _Metric(M), _Metric(F))
    row = judge(pipe)

    assert set(row["metric_results"]) == {M, F}


def test_two_crashes_are_both_reported(tmp_path):
    pipe = make_pipeline(tmp_path, _Metric(R, boom=True), _Metric(M), _Metric(F, boom=True))
    row = judge(pipe)

    assert row["query_status"] == QueryStatus.JUDGED
    assert row["last_error"].count("; ") == 1
    assert "recall_at_k:" in row["last_error"] and "faithfulness:" in row["last_error"]


def test_every_metric_crashing_keeps_the_old_failed_row(tmp_path):                      # D8-Q29
    pipe = make_pipeline(tmp_path, _Metric(R, boom=True), _Metric(F, boom=True))
    row = judge(pipe)

    assert row["query_status"] == QueryStatus.FAILED
    assert row["failure_mode"] == FailureMode.FAILED
    assert "all metrics failed" in row["last_error"]
    assert "recall_at_k" in row["last_error"] and "faithfulness" in row["last_error"]
    assert "metric_results" not in row or not row["metric_results"]
    aggregate(pipe, row)
    assert row["failure_mode"] == FailureMode.FAILED                                    # _aggregate leaves FAILED rows alone


# ---------------------------------------------------------------- _aggregate

def test_missing_metric_is_tagged_judge_failed(tmp_path):
    pipe = make_pipeline(tmp_path, _Metric(R), _Metric(F, boom=True))
    row = judge(pipe)
    aggregate(pipe, row)

    assert row["failure_mode"] == FailureMode.JUDGE_FAILED
    assert str(FailureMode.JUDGE_FAILED) == "judge_failed"


def test_judge_failed_overrides_a_retrieval_miss(tmp_path):                             # D8-Q30
    pipe = make_pipeline(tmp_path, _Metric(R, 0.0), _Metric(F, boom=True))
    row = judge(pipe)
    aggregate(pipe, row)

    assert row["failure_mode"] == FailureMode.JUDGE_FAILED


def test_run_level_means_skip_the_missing_metric_only(tmp_path):
    good = make_pipeline(tmp_path, _Metric(R, 1.0), _Metric(F, 1.0))
    bad = make_pipeline(tmp_path, _Metric(R, 0.5), _Metric(F, boom=True))
    row_ok, row_bad = judge(good, "q1"), judge(bad, "q2")

    agg = aggregate(good, row_ok, row_bad)            # `good` knows both metric names

    assert agg[R] == pytest.approx(0.75)              # both rows count for recall
    assert agg[F] == pytest.approx(1.0)               # only the complete row counts for faithfulness
    assert row_bad["failure_mode"] == FailureMode.JUDGE_FAILED


# ---------------------------------------------------------------- storage + comparator

def test_persisted_judge_failed_row_is_unmeasured_only_for_the_missing_metric(tmp_path):
    pipe = make_pipeline(tmp_path, _Metric(R, 0.5), _Metric(F, 1.0))
    ok_row = judge(pipe, "q1")
    bad_pipe = make_pipeline(tmp_path, _Metric(R, 0.5), _Metric(F, boom=True))
    bad_row = judge(bad_pipe, "q2")
    aggregate(pipe, ok_row, bad_row)

    Repository(pipe.db_path).create_run(
        Run(run_id="r1", created_at="2026-10-21T00:00:00", is_dev=False, k=5, max_workers=1,
            config={"is_dev": False, "k": 5}, config_hash=None))
    pipe._persist({"run_id": "r1", "per_query_results": [ok_row, bad_row], "aggregated_metrics": {R: 0.5, F: 1.0}})

    stored = {r.qa_id: r for r in Repository(pipe.db_path).get_query_results("r1")}
    assert stored["q2"].query_status == "judged"
    assert stored["q2"].failure_mode == "judge_failed"
    assert "faithfulness" in stored["q2"].last_error
    assert set(stored["q2"].metric_results) == {R}

    snap = load_snapshot(Repository(pipe.db_path), "r1")
    row = snap.rows["q2"]
    assert row.kind is RowKind.MEASURED
    assert row.values[R] == pytest.approx(0.5)
    assert row.values[F] is None                                                        # excluded from the faithfulness delta

    validity = run_validity(snap, [R, F])
    assert (validity.measured, validity.attempted) == (1, 2)                           # lowers validity, as designed (D8-Q8)