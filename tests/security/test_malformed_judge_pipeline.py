"""
tests/security/test_malformed_judge_pipeline.py

a judge that misbehaves, seen through the COMPILED graph, the real
Faithfulness metric, SQLite storage and the comparator's validity count. No Ollama, no models.

One 5-question run (fixture qa_5doc.json), one judge behaviour per question:

    row 0  ok          1 claim, verdict 1                   -> faithfulness 1.0
    row 1  no_claims   judge extracts no claims             -> NaN "No claims extracted."
    row 2  mismatch    2 claims, 1 verdict                  -> NaN "judge returned 1 verdicts for 2 statements"
    row 3  raise       judge raises                         -> metric absent, row `judge_failed`
    row 4  ok

Pinned behaviour (D11-Q10 = A, F-D11-14 / F-D11-15): a NaN faithfulness row is stored as
`judged` and tagged `pass` (the metric is present, only the LOW_FAITHFULNESS branch is
skipped). What protects the comparison is that the comparator reads NaN as UNMEASURED, so
validity drops (-> INCONCLUSIVE below 80 %). Both facts are asserted. If the row tag ever
becomes `judge_failed` for NaN, update the two tagged assertions deliberately.

Safety (F-D10-33): temp SQLite only.

Run (repo root):  python -m pytest tests/security/test_malformed_judge_pipeline.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import QAPair
from ragregress.evaluation.comparator import RowKind, load_snapshot
from ragregress.evaluation.metrics import MRR, Faithfulness, RecallAtK, TokenPRF1
from ragregress.evaluation.metrics.faithfulness.utils import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)
from ragregress.evaluation.regression import run_validity
from ragregress.pipeline.pipeline import Pipeline
from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.storage.repositories import Repository

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QA = [QAPair(**q) for q in json.loads((FIXTURES / "qa_5doc.json").read_text(encoding="utf-8"))]
BY_QUESTION = {q.question: q for q in QA}
RUN_ID = "fx-judge"

MODES = ["ok", "no_claims", "mismatch", "raise", "ok"]          # one per fixture row
MODE_OF = {qa.qa_id: mode for qa, mode in zip(QA, MODES)}


# ---------------------------------------------------------------- fakes

class FakeRag:
    """Gold passage at rank 1, first gold answer: every non-judge metric is perfect."""

    def retrieve(self, question, k=5):
        if question == "health check probe":
            return []
        gold = BY_QUESTION[question].gold_passage_ids[0]
        return [Chunk(chunk_id=f"{gold}::0", source_passage_id=gold, text="gold text")]

    def generate(self, question, contexts):
        answer = BY_QUESTION[question].answer[0]
        return GenerationResult(raw_output=answer, answer=answer, citations=[])


class PerQuestionJudge:
    """A StructuredLLM whose behaviour depends on which question is being judged.

    Claims step: the prompt contains the question, so the mode is looked up from it.
    NLI step: the prompt lists the claims, whose text carries the mode tag.
    """

    def __init__(self, modes_by_question):
        self.modes_by_question = modes_by_question

    def generate_structured(self, prompt, schema):
        if schema is StatementGeneratorOutput:
            mode = next(m for q, m in self.modes_by_question.items() if q in prompt)
            if mode == "raise":
                raise RuntimeError("judge exploded")
            claims = {"ok": ["ok claim"], "no_claims": [], "mismatch": ["mismatch claim a", "mismatch claim b"]}[mode]
            return StatementGeneratorOutput(statements=claims)
        if schema is NLIStatementOutput:
            claim = "mismatch claim a" if "mismatch claim" in prompt else "ok claim"
            return NLIStatementOutput(statements=[
                StatementFaithfulnessAnswer(statement=claim, reason="r", verdict=1)])
        raise AssertionError(f"unexpected schema {schema!r}")


def run_pipeline(tmp_path, modes):
    judge = PerQuestionJudge({qa.question: m for qa, m in zip(QA, modes)})
    metrics = [RecallAtK(k=5), MRR(k=5), TokenPRF1(), Faithfulness(llm=judge)]
    pipe = Pipeline(rag_system=FakeRag(), metrics=metrics, k=5, max_workers=1, db_path=tmp_path / "pipe.db")
    pipe.run(qa_pairs=QA, run_id=RUN_ID, is_dev=False)
    repo = Repository(pipe.db_path)
    rows = {r.qa_id: r for r in repo.get_query_results(RUN_ID)}
    agg = {m.metric_name: m.value for m in repo.get_metrics(RUN_ID)}
    return repo, repo.get_run(RUN_ID), rows, agg


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    return run_pipeline(tmp_path_factory.mktemp("mixed"), MODES)


def row_of(rows, mode, nth=0):
    ids = [qa.qa_id for qa in QA if MODE_OF[qa.qa_id] == mode]
    return rows[ids[nth]]


# ---------------------------------------------------------------- the run survives

def test_run_stays_healthy_and_stores_every_row(mixed):
    _, run_row, rows, _ = mixed
    assert run_row.run_status == "healthy"
    assert len(rows) == 5
    assert {r.query_status for r in rows.values()} == {"judged"}


def test_well_formed_rows_pass_with_a_faithfulness_score(mixed):
    _, _, rows, _ = mixed
    for nth in (0, 1):
        row = row_of(rows, "ok", nth)
        assert row.failure_mode == "pass"
        assert row.metric_results["faithfulness"]["value"] == pytest.approx(1.0)


# ---------------------------------------------------------------- judge raises

def test_raising_judge_gives_judge_failed_without_the_metric(mixed):
    _, _, rows, _ = mixed
    row = row_of(rows, "raise")
    assert row.failure_mode == "judge_failed"
    assert "faithfulness" not in row.metric_results
    assert set(row.metric_results) == {"recall_at_k", "mrr", "token_prf1"}
    assert "judge exploded" in row.last_error


# ---------------------------------------------------------------- NaN rows (pinned, D11-Q10 = A)

@pytest.mark.parametrize(
    "mode, reason",
    [("no_claims", "No claims extracted."),
     ("mismatch", "judge returned 1 verdicts for 2 statements")],
)
def test_nan_faithfulness_row_is_stored_with_its_reason(mixed, mode, reason):
    _, _, rows, _ = mixed
    stored = row_of(rows, mode).metric_results["faithfulness"]
    assert math.isnan(stored["value"])
    assert stored["reason"] == reason


@pytest.mark.parametrize("mode", ["no_claims", "mismatch"])
def test_nan_faithfulness_row_is_tagged_pass_today(mixed, mode):
    """PINNED current behaviour (F-D11-14): misleading tag, protected by the validity count below."""
    _, _, rows, _ = mixed
    row = row_of(rows, mode)
    assert (row.query_status, row.failure_mode) == ("judged", "pass")


def test_nan_and_missing_faithfulness_are_left_out_of_the_run_mean(mixed):
    _, _, _, agg = mixed
    assert agg["faithfulness"] == pytest.approx(1.0)           # mean of the two well-formed rows only
    assert agg["recall_at_k"] == pytest.approx(1.0)


# ---------------------------------------------------------------- what protects the comparison

def test_comparator_counts_nan_and_missing_faithfulness_as_unmeasured(mixed):
    repo, _, _, _ = mixed
    snap = load_snapshot(repo, RUN_ID)
    by_mode = {mode: snap.rows[qa.qa_id] for qa, mode in zip(QA, MODES)}

    assert by_mode["no_claims"].values["faithfulness"] is None
    assert by_mode["mismatch"].values["faithfulness"] is None
    assert by_mode["raise"].values["faithfulness"] is None
    assert by_mode["ok"].values["faithfulness"] == pytest.approx(1.0)
    assert by_mode["raise"].kind is RowKind.MEASURED


def test_validity_drops_below_the_gate_when_faithfulness_is_gated(mixed):
    repo, _, _, _ = mixed
    snap = load_snapshot(repo, RUN_ID)

    gated_on_faithfulness = run_validity(snap, ["faithfulness"])
    assert (gated_on_faithfulness.measured, gated_on_faithfulness.attempted) == (2, 5)
    assert not gated_on_faithfulness.is_valid()                # 40 % < 80 % -> compare says INCONCLUSIVE

    gated_on_recall = run_validity(snap, ["recall_at_k"])
    assert (gated_on_recall.measured, gated_on_recall.attempted) == (5, 5)


# ---------------------------------------------------------------- every row NaN

def test_all_nan_faithfulness_stores_a_null_aggregate_and_zero_validity(tmp_path):
    repo, run_row, rows, agg = run_pipeline(tmp_path, ["no_claims"] * 5)

    assert run_row.run_status == "healthy"
    assert {r.failure_mode for r in rows.values()} == {"pass"}          # pinned, see module docstring
    assert agg["faithfulness"] is None                                  # NaN mean stored as NULL
    assert agg["recall_at_k"] == pytest.approx(1.0)

    validity = run_validity(load_snapshot(repo, RUN_ID), ["faithfulness"])
    assert (validity.measured, validity.attempted) == (0, 5)
    assert not validity.is_valid()