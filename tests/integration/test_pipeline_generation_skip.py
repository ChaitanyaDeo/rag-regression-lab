"""
tests/integration/test_pipeline_generation_skip.py

Generation runs only when a selected metric needs the generated `answer`.
A metric that needs only `context` (retrieved chunks) must not trigger the generator.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ragregress.benchmark.schemas import QAPair
from ragregress.evaluation.metrics.base import MetricResult
from ragregress.pipeline.pipeline import Pipeline
from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.storage.repositories import Repository

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QA = [QAPair(**q) for q in json.loads((FIXTURES / "qa_5doc.json").read_text(encoding="utf-8"))]
BY_QUESTION = {q.question: q for q in QA}
RUN_ID = "gen-skip"


class CountingRag:
    def __init__(self):
        self.generate_calls = 0
        self._lock = threading.Lock()

    def retrieve(self, question, k=5):
        if question == "health check probe":
            return []
        gold = BY_QUESTION[question].gold_passage_ids[0]
        return [Chunk(chunk_id=f"{gold}::0", source_passage_id=gold, text="gold text")]

    def generate(self, question, contexts):
        with self._lock:
            self.generate_calls += 1
        answer = BY_QUESTION[question].answer[0]
        return GenerationResult(raw_output=answer, answer=answer, citations=[])


class ContextOnlyMetric:
    name = "context_only"
    required_inputs = frozenset({"question", "reference", "context"})

    def __init__(self):
        self.seen = []

    def score(self, *, question, reference, context):
        self.seen.append((question, [c.source_passage_id for c in context]))
        return MetricResult(value=1.0)


class AnswerMetric:
    name = "answer_metric"
    required_inputs = frozenset({"answer"})

    def score(self, *, answer):
        return MetricResult(value=1.0)


def run(tmp_path, rag, metrics):
    pipe = Pipeline(rag_system=rag, metrics=metrics, k=5, db_path=tmp_path / "pipe.db")
    pipe.run(qa_pairs=QA, run_id=RUN_ID, is_dev=False)
    return {r.qa_id: r for r in Repository(pipe.db_path).get_query_results(RUN_ID)}


def test_context_only_metric_does_not_call_the_generator(tmp_path):
    rag, metric = CountingRag(), ContextOnlyMetric()
    rows = run(tmp_path, rag, [metric])

    assert rag.generate_calls == 0
    assert len(rows) == len(QA)
    for q in QA:
        r = rows[q.qa_id]
        assert r.query_status == "judged"
        assert r.generation_answer is None
        assert "generate" not in r.latency
    # the metric still received the retrieved chunks
    assert sorted(metric.seen) == sorted((q.question, [q.gold_passage_ids[0]]) for q in QA)


def test_answer_metric_still_triggers_generation(tmp_path):
    rag = CountingRag()
    rows = run(tmp_path, rag, [ContextOnlyMetric(), AnswerMetric()])

    assert rag.generate_calls == len(QA)
    for q in QA:
        assert rows[q.qa_id].generation_answer == q.answer[0]
        assert "generate" in rows[q.qa_id].latency