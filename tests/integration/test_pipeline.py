"""
tests/integration/test_pipeline.py

5-doc / 5-query integration test (D10-B7, D10-Q3): BuiltinRAGSystem (real chunker, real MiniLM,
real Chroma in a TEMP folder) + Pipeline.run() on tests/fixtures, run TWICE, stored results compared.

Stubbed part (default, D10-Q22 option A): ChatOllama is replaced by a fake that answers with the
fixture's gold answer plus a [source: pid] citation, and the real Faithfulness metric gets a fake
judge LLM -- so prompt building, citation parsing and faithfulness scoring still run for real.

Real part set RAGREGRESS_SLOW=1. Uses configs/baseline.yaml (3B generator,
7B judge, seeds 42) through real Ollama; ~4-5 min for the two runs. Run from the repo root.

Safety (F-D10-33): the collection name depends only on embedding model + chunk settings, so indexing
the fixture into the default artifacts/chroma would replace the real baseline index. Every test here
uses a temp persist_directory.

Run (repo root):  python -m pytest tests/integration/test_pipeline.py -v
       real part:  $env:RAGREGRESS_SLOW=1; python -m pytest tests/integration/test_pipeline.py -v
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage, QAPair
from ragregress.evaluation.metrics import MRR, Faithfulness, RecallAtK, TokenPRF1
from ragregress.evaluation.metrics.faithfulness.utils import (
    NLIStatementOutput, StatementFaithfulnessAnswer, StatementGeneratorOutput,
)
from ragregress.pipeline.pipeline import Pipeline
from ragregress.rag import generator as generator_mod
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.rag.generator import GeneratorConfig
from ragregress.storage.repositories import Repository

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


PASSAGES = [Passage(**p) for p in load_fixture("passages_5doc.json")]
QA = [QAPair(**q) for q in load_fixture("qa_5doc.json")]
BY_QUESTION = {q.question: q for q in QA}
PASSAGE_IDS = {p.passage_id for p in PASSAGES}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SLOW = os.environ.get("RAGREGRESS_SLOW") == "1"


# ---------------------------------------------------------------- helpers shared by both parts

def build_system(tmp_path, chunk_size=500, overlap=50, generator_config=None, embedding_model=EMBEDDING_MODEL):
    system = BuiltinRAGSystem(ChunkerConfig(chunk_size=chunk_size, overlap=overlap),
                              generator_config=generator_config or GeneratorConfig(),
                              embedding_model=embedding_model,
                              persist_directory=str(tmp_path / "chroma"))
    system.ensure_indexed(PASSAGES)
    return system


def run_twice(system, metrics, tmp_path, k=5):
    """Run the 5 fixture queries twice into one temp DB; return (state, rows, aggregates) per run."""
    db = tmp_path / "pipeline.db"
    out = []
    for run_id in ("run-1", "run-2"):
        pipe = Pipeline(rag_system=system, metrics=metrics, k=k, db_path=db)
        state = pipe.run(qa_pairs=QA, run_id=run_id, is_dev=False)
        repo = Repository(db)
        rows = {r.qa_id: r for r in repo.get_query_results(run_id)}
        agg = {m.metric_name: m.value for m in repo.get_metrics(run_id)}
        out.append((state, repo.get_run(run_id), rows, agg))
    return out


def comparable(rows, agg):
    """Everything stored except timings."""
    row_view = {
        qa_id: (r.query_status, r.failure_mode, r.last_error, r.retrieved_ids, r.generation_answer,
                {name: res.get("value") for name, res in r.metric_results.items()})
        for qa_id, r in rows.items()
    }
    return row_view, {k: v for k, v in agg.items() if not k.startswith("latency")}


# ---------------------------------------------------------------- stubbed part (default)

class FakeChatOllama:
    """Reads the real generation prompt; answers with the gold answer + a citation to the first
    fixture passage present in the context block."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, prompt_value):
        prompt = prompt_value.to_string()
        question = re.findall(r"^Question: (.*)$", prompt, flags=re.MULTILINE)[-1]   # last = the real one
        cited = next(pid for pid in re.findall(r"\[source: (p\d+)\]", prompt) if pid in PASSAGE_IDS)
        answer = BY_QUESTION[question].answer[0]
        return type("Msg", (), {"content": f"{answer} [source: {cited}]"})()


class FakeJudge:
    """One statement per answer; supported (1) when the context block carries a fixture passage tag."""

    def generate_structured(self, prompt, schema):
        if schema is StatementGeneratorOutput:
            answer = prompt.split("Answer:")[-1].split("Statements:")[0].strip()
            return StatementGeneratorOutput(statements=[answer])
        if schema is NLIStatementOutput:
            supported = int(any(f"[{pid}]" in prompt for pid in PASSAGE_IDS))
            statement = re.findall(r"^1\. (.*)$", prompt, flags=re.MULTILINE)[-1]
            return NLIStatementOutput(statements=[
                StatementFaithfulnessAnswer(statement=statement, reason="fake", verdict=supported)])
        raise AssertionError(f"unexpected schema {schema!r}")


@pytest.fixture(scope="module")
def stubbed_runs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(generator_mod, "ChatOllama", FakeChatOllama)
    try:
        tmp = tmp_path_factory.mktemp("pipeline_stubbed")
        system = build_system(tmp)
        metrics = [RecallAtK(k=5), MRR(k=5), TokenPRF1(), Faithfulness(llm=FakeJudge())]
        yield tmp, run_twice(system, metrics, tmp)
    finally:
        mp.undo()


def test_index_lives_in_the_temp_folder(stubbed_runs):
    tmp, _ = stubbed_runs
    assert (tmp / "chroma" / "chroma.sqlite3").is_file()


def test_run_is_healthy_and_stores_all_five_rows(stubbed_runs):
    _, ((state, run_row, rows, _), _) = stubbed_runs
    assert run_row.run_status == "healthy"
    assert set(rows) == {q.qa_id for q in QA}


def test_every_row_passes_with_perfect_scores(stubbed_runs):
    _, ((_, _, rows, agg), _) = stubbed_runs
    for r in rows.values():
        assert (r.query_status, r.failure_mode) == ("judged", "pass")
        assert {n: v["value"] for n, v in r.metric_results.items()} == pytest.approx(
            {"recall_at_k": 1.0, "mrr": 1.0, "token_prf1": 1.0, "faithfulness": 1.0})
    assert {k: v for k, v in agg.items() if not k.startswith("latency")} == pytest.approx(
        {"recall_at_k": 1.0, "mrr": 1.0, "token_prf1": 1.0, "faithfulness": 1.0})


def test_answers_are_stored_without_citations_and_ids_are_real(stubbed_runs):
    _, ((_, _, rows, _), _) = stubbed_runs
    for q in QA:
        r = rows[q.qa_id]
        assert r.generation_answer == q.answer[0]
        assert "[source:" not in r.generation_answer
        assert r.retrieved_ids[0] == q.gold_passage_ids[0]
        assert set(r.retrieved_ids) <= PASSAGE_IDS


def test_latency_is_recorded_for_every_stage(stubbed_runs):
    _, ((_, _, rows, agg), _) = stubbed_runs
    assert all(set(r.latency) == {"retrieve", "generate", "judge"} for r in rows.values())
    assert not math.isnan(agg["latency_p50_total"])


def test_two_runs_store_identical_results(stubbed_runs):
    _, ((_, _, rows1, agg1), (_, _, rows2, agg2)) = stubbed_runs
    assert comparable(rows1, agg1) == comparable(rows2, agg2)


# ---------------------------------------------------------------- real part (opt-in)

@pytest.mark.skipif(not SLOW, reason="real Ollama run; set RAGREGRESS_SLOW=1")
def test_real_models_run_twice_identically(tmp_path):
    from ragregress.core.config import load_run_config
    from ragregress.core.runner import build_metrics

    cfg = load_run_config("configs/baseline.yaml")
    p = cfg.pipeline.params
    system = build_system(tmp_path, chunk_size=p.chunk_size, overlap=p.overlap, embedding_model=p.embedding_model,
                          generator_config=GeneratorConfig(model_name=p.generation_model, seed=p.seed))
    (_, run1, rows1, agg1), (_, run2, rows2, agg2) = run_twice(system, build_metrics(cfg), tmp_path, k=cfg.top_k)

    assert run1.run_status == run2.run_status == "healthy"
    assert len(rows1) == len(rows2) == 5
    for q in QA:
        assert rows1[q.qa_id].retrieved_ids[0] == q.gold_passage_ids[0]
    assert comparable(rows1, agg1) == comparable(rows2, agg2)