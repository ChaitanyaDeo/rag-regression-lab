"""
tests/integration/test_pipeline_failures.py

Failure handling through the COMPILED LangGraph graph: Pipeline.run()
end to end on the 5-doc fixture, temp SQLite, fake RAG system, real retrieval/F1 metrics plus a
stub "faithfulness" judge. No Ollama, no models.

Covers: judge failure on one row / every metric / every row; retrieve and generate retry -> fail
routing (MAX_RETRIES = 3); recovery after transient retrieve errors; unhealthy run at the ingest
health probe; progress callback; max_workers=4 gives the same stored outcome as sequential.

(test_pipeline_judge.py calls _judge / _aggregate / _persist directly; this file checks that the
graph wiring delivers the same behaviour.)

Run (repo root):  python -m pytest tests/integration/test_pipeline_failures.py -v
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import QAPair
from ragregress.evaluation.metrics import MRR, RecallAtK, TokenPRF1
from ragregress.evaluation.metrics.base import MetricResult
from ragregress.pipeline.pipeline import MAX_RETRIES, Pipeline
from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.storage.repositories import Repository

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QA = [QAPair(**q) for q in json.loads((FIXTURES / "qa_5doc.json").read_text(encoding="utf-8"))]
BY_QUESTION = {q.question: q for q in QA}
Q0, Q1 = QA[0], QA[1]
RUN_ID = "fx-run"


# ---------------------------------------------------------------- fakes

class FakeRag:
    """Retrieves the gold passage at rank 1 and 'generates' the first gold answer.

    fail_retrieve: {question: n} -> the first n retrieve calls for that question raise
    (use a large n for 'always'). fail_generate: same for generate. probe_fails: the
    ingest health probe raises.
    """

    def __init__(self, fail_retrieve=None, fail_generate=None, probe_fails=False):
        self.fail_retrieve = dict(fail_retrieve or {})
        self.fail_generate = dict(fail_generate or {})
        self.probe_fails = probe_fails
        self.calls = Counter()
        self._lock = threading.Lock()

    def _maybe_fail(self, kind, question, budget):
        with self._lock:
            self.calls[(kind, question)] += 1
            n = self.calls[(kind, question)]
        if n <= budget.get(question, 0):
            raise RuntimeError(f"{kind} boom #{n}")

    def retrieve(self, question, k=5):
        if question == "health check probe":
            if self.probe_fails:
                raise ConnectionError("vector store unreachable")
            return []
        self._maybe_fail("retrieve", question, self.fail_retrieve)
        gold = BY_QUESTION[question].gold_passage_ids[0]
        return [Chunk(chunk_id=f"{gold}::0", source_passage_id=gold, text="gold text")]

    def generate(self, question, contexts):
        self._maybe_fail("generate", question, self.fail_generate)
        answer = BY_QUESTION[question].answer[0]
        return GenerationResult(raw_output=answer, answer=answer, citations=[])


class StubFaithfulness:
    """Named like the real metric so _aggregate's failure-mode chain applies; crashes on chosen questions."""
    name = "faithfulness"
    required_inputs = frozenset({"question", "answer", "context"})

    def __init__(self, crash_on=(), crash_all=False):
        self.crash_on, self.crash_all = set(crash_on), crash_all

    def score(self, *, question, answer, context):
        if self.crash_all or question in self.crash_on:
            raise RuntimeError("judge exploded")
        return MetricResult(value=1.0)


class CrashingMetric:
    """Any metric that always raises (used to make every metric fail on a row)."""
    def __init__(self, name, crash_on):
        self.name, self.crash_on = name, set(crash_on)
        self.required_inputs = frozenset({"question"})

    def score(self, *, question):
        if question in self.crash_on:
            raise RuntimeError(f"{self.name} exploded")
        return MetricResult(value=1.0)


def metrics(faith=None):
    return [RecallAtK(k=5), MRR(k=5), TokenPRF1(), faith or StubFaithfulness()]


def run(tmp_path, rag, metric_list=None, max_workers=1, on_row_done=None):
    pipe = Pipeline(rag_system=rag, metrics=metric_list or metrics(), k=5,
                    max_workers=max_workers, db_path=tmp_path / "pipe.db")
    pipe.on_row_done = on_row_done
    state = pipe.run(qa_pairs=QA, run_id=RUN_ID, is_dev=False)
    repo = Repository(pipe.db_path)
    rows = {r.qa_id: r for r in repo.get_query_results(RUN_ID)}
    stored_metrics = {m.metric_name: m.value for m in repo.get_metrics(RUN_ID)}
    return state, repo.get_run(RUN_ID), rows, stored_metrics


def modes(rows):
    return {qa_id: r.failure_mode for qa_id, r in rows.items()}


# ---------------------------------------------------------------- judge failures

def test_judge_failure_does_not_abort_run(tmp_path):
    _, run_row, rows, agg = run(tmp_path, FakeRag(), metrics(StubFaithfulness(crash_on={Q0.question})))

    assert run_row.run_status == "healthy"
    assert len(rows) == 5
    bad = rows[Q0.qa_id]
    assert (bad.query_status, bad.failure_mode) == ("judged", "judge_failed")
    assert "faithfulness" in bad.last_error and "judge exploded" in bad.last_error
    assert set(bad.metric_results) == {"recall_at_k", "mrr", "token_prf1"}
    assert all(rows[q.qa_id].failure_mode == "pass" for q in QA[1:])
    assert agg["faithfulness"] == pytest.approx(1.0) and agg["recall_at_k"] == pytest.approx(1.0)


def test_every_metric_crashing_on_one_row_fails_only_that_row(tmp_path):
    crash = {Q0.question}
    metric_list = [CrashingMetric("recall_at_k", crash), CrashingMetric("faithfulness", crash)]
    _, run_row, rows, _ = run(tmp_path, FakeRag(), metric_list)

    assert run_row.run_status == "healthy"
    assert (rows[Q0.qa_id].query_status, rows[Q0.qa_id].failure_mode) == ("failed", "failed")
    assert "all metrics failed" in rows[Q0.qa_id].last_error
    assert all(rows[q.qa_id].query_status == "judged" for q in QA[1:])


def test_judge_crashing_on_every_row_still_persists_the_run(tmp_path):
    _, run_row, rows, agg = run(tmp_path, FakeRag(), metrics(StubFaithfulness(crash_all=True)))

    assert run_row.run_status == "healthy"
    assert set(modes(rows).values()) == {"judge_failed"}
    assert agg["faithfulness"] is None                     # NaN mean stored as NULL
    assert agg["recall_at_k"] == pytest.approx(1.0)


# ---------------------------------------------------------------- retrieve / generate routing

def test_retrieve_always_failing_ends_failed_after_max_retries(tmp_path):
    rag = FakeRag(fail_retrieve={Q0.question: 99})
    _, run_row, rows, _ = run(tmp_path, rag)

    assert run_row.run_status == "healthy"
    assert rag.calls[("retrieve", Q0.question)] == MAX_RETRIES == 3
    assert (rows[Q0.qa_id].query_status, rows[Q0.qa_id].failure_mode) == ("failed", "failed")
    assert "retrieve boom #3" in rows[Q0.qa_id].last_error
    assert all(rows[q.qa_id].failure_mode == "pass" for q in QA[1:])


def test_transient_retrieve_errors_recover(tmp_path):
    rag = FakeRag(fail_retrieve={Q1.question: 2})
    _, _, rows, _ = run(tmp_path, rag)

    assert rag.calls[("retrieve", Q1.question)] == 3
    assert (rows[Q1.qa_id].query_status, rows[Q1.qa_id].failure_mode) == ("judged", "pass")


def test_generate_always_failing_ends_failed_after_max_retries(tmp_path):
    rag = FakeRag(fail_generate={Q0.question: 99})
    _, run_row, rows, _ = run(tmp_path, rag)

    assert run_row.run_status == "healthy"
    assert rag.calls[("generate", Q0.question)] == MAX_RETRIES
    assert rag.calls[("retrieve", Q0.question)] == 1            # retrieval is not repeated
    assert (rows[Q0.qa_id].query_status, rows[Q0.qa_id].failure_mode) == ("failed", "failed")
    assert rows[Q0.qa_id].generation_answer is None
    assert "generate boom #3" in rows[Q0.qa_id].last_error


# ---------------------------------------------------------------- run level

def test_failed_health_probe_marks_run_unhealthy_with_no_rows(tmp_path):
    rag = FakeRag(probe_fails=True)
    state, run_row, rows, _ = run(tmp_path, rag)

    assert run_row.run_status == "unhealthy"
    assert "vector store unreachable" in run_row.run_status_detail
    assert rows == {}
    assert not any(kind == "retrieve" for kind, _ in rag.calls)  # no query branch started


def test_progress_callback_counts_failed_rows_too(tmp_path):
    calls = []
    rag = FakeRag(fail_retrieve={Q0.question: 99}, fail_generate={Q1.question: 99})
    run(tmp_path, rag, on_row_done=lambda done, total, elapsed: calls.append((done, total)))

    assert sorted(d for d, _ in calls) == [0, 1, 2, 3, 4, 5]
    assert {t for _, t in calls} == {5}


def test_parallel_run_stores_the_same_outcome_as_sequential(tmp_path):
    def outcome(sub, workers):
        rag = FakeRag(fail_retrieve={Q0.question: 99}, fail_generate={Q1.question: 2})
        _, run_row, rows, agg = run(tmp_path / sub, rag, metrics(StubFaithfulness(crash_on={QA[2].question})),
                                    max_workers=workers)
        return run_row.run_status, modes(rows), {k: v for k, v in agg.items() if not k.startswith("latency")}

    (tmp_path / "seq").mkdir()
    (tmp_path / "par").mkdir()
    seq, par = outcome("seq", 1), outcome("par", 4)
    assert seq == par
    assert seq[1] == {QA[0].qa_id: "failed", QA[1].qa_id: "pass", QA[2].qa_id: "judge_failed",
                      QA[3].qa_id: "pass", QA[4].qa_id: "pass"}