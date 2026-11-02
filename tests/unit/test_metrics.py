"""
tests/unit/test_metrics.py

Per-row metric logic: RecallAtK, MRR, TokenPRF1 and the deterministic parts of
Faithfulness (score arithmetic, NaN paths, input checks) with a fake judge -- no Ollama.

Pinned behaviour: retrieved ids are per CHUNK, so one passage can appear several
times. Recall@K counts a gold id once (set); MRR ranks over the chunk list, so duplicate
non-gold chunks ahead of the gold push its rank down. Changing that would move the
canonical numbers -- it would be a deliberate new metric version, not a fix.

Run (repo root):  python -m pytest tests/unit/test_metrics.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage, QAPair
from ragregress.evaluation.metrics import (
    METRIC_REGISTRY, MRR, Faithfulness, Metric, RecallAtK, TokenPRF1,
)
from ragregress.evaluation.metrics.faithfulness.utils import (
    NLIStatementOutput, StatementFaithfulnessAnswer, StatementGeneratorOutput,
)
from ragregress.rag.interface import Chunk

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


PASSAGES = [Passage(**p) for p in load_fixture("passages_5doc.json")]
QA = [QAPair(**q) for q in load_fixture("qa_5doc.json")]
A, B, C = (p.passage_id for p in PASSAGES[:3])   # 3-doc fixture ids
OTHER = PASSAGES[3].passage_id


def recall(retrieved, gold, k=5):
    return RecallAtK(k=k).score(retrieved_ids=retrieved, gold_ids=gold).value


def mrr(retrieved, gold, k=5):
    return MRR(k=k).score(retrieved_ids=retrieved, gold_ids=gold).value


def f1(answer, gold):
    return TokenPRF1().score(answer=answer, gold_answers=gold)


# ---------------------------------------------------------------- Recall@K

@pytest.mark.parametrize("retrieved,expected", [
    ([A, B, C], 1.0),          # gold at rank 1
    ([B, C, A], 1.0),          # gold at rank k (k=3)
    ([B, C, OTHER, A], 0.0),   # gold just beyond k
    ([], 0.0),                 # nothing retrieved
])
def test_recall_single_gold(retrieved, expected):
    assert recall(retrieved, [A], k=3) == expected


def test_recall_empty_gold_is_zero():
    assert recall([A, B], []) == 0.0


def test_recall_two_golds_one_found_is_half():
    assert recall([A, C], [A, B]) == 0.5


def test_recall_duplicate_chunks_of_gold_count_once():
    assert recall([A, A, A], [A, B]) == 0.5


# ---------------------------------------------------------------- MRR

@pytest.mark.parametrize("retrieved,expected", [
    ([A, B, C], 1.0),
    ([B, A, C], 0.5),
    ([B, C, A], 1 / 3),
    ([B, C, OTHER, A], 0.0),   # beyond k=3
    ([], 0.0),
])
def test_mrr_rank(retrieved, expected):
    assert mrr(retrieved, [A], k=3) == pytest.approx(expected)


def test_mrr_first_of_several_golds_wins():
    assert mrr([C, B, A], [A, B]) == 0.5


def test_mrr_duplicate_non_gold_chunks_push_rank_down():
    # ranks count chunks, not distinct passages
    assert mrr([B, B, A], [A]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- TokenPRF1 (SQuAD F1)

def test_f1_identical_is_one():
    assert f1("Roger Goodell", ["Roger Goodell"]).value == 1.0


def test_f1_disjoint_is_zero():
    r = f1("Belgrade", ["Roger Goodell"])
    assert (r.value, r.details) == (0.0, {"precision": 0.0, "recall": 0.0})


def test_f1_partial_overlap_hand_computed():
    # pred [cat, sat]; gold [cat, sat, on, mat] after dropping "the"/"a" -> P=1, R=0.5, F1=2/3
    r = f1("cat sat", ["the cat sat on a mat"])
    assert r.value == pytest.approx(2 / 3)
    assert r.details == pytest.approx({"precision": 1.0, "recall": 0.5})


def test_f1_counts_repeated_tokens_as_a_multiset():
    # pred [cat, cat] vs gold [cat]: common = 1 -> P=0.5, R=1, F1=2/3
    assert f1("the cat cat", ["cat"]).value == pytest.approx(2 / 3)
    # pred [cat, cat] vs gold [cat, cat, dog]: common = 2 -> P=1, R=2/3, F1=0.8
    assert f1("cat cat", ["cat cat dog"]).value == pytest.approx(0.8)


def test_f1_ignores_case_punctuation_and_articles():
    assert f1("The Rankine Cycle!", ["rankine cycle"]).value == 1.0


def test_f1_takes_best_gold_answer():
    # fixture row 4 golds: "Roger Goodell" (F1 2/3), "Goodell" (F1 0.4)
    gold = next(q.answer for q in QA if q.qa_id.startswith("56be53b8"))
    assert f1("NFL Commissioner Roger Goodell", gold).value == pytest.approx(2 / 3)


@pytest.mark.parametrize("answer,expected", [("", 1.0), ("something", 0.0)])
def test_f1_no_answer_gold(answer, expected):
    assert f1(answer, [""]).value == expected


def test_f1_empty_gold_list_raises():
    with pytest.raises(ValueError, match="gold_answers must be non-empty"):
        f1("x", [])


# ---------------------------------------------------------------- Faithfulness (fake judge)

class FakeJudge:
    """Returns canned statements / verdicts and records the prompts it was given."""

    def __init__(self, statements, verdicts):
        self.statements, self.verdicts, self.prompts = statements, verdicts, []

    def generate_structured(self, prompt, schema):
        self.prompts.append(prompt)
        if schema is StatementGeneratorOutput:
            return StatementGeneratorOutput(statements=self.statements)
        if schema is NLIStatementOutput:
            return NLIStatementOutput(statements=[
                StatementFaithfulnessAnswer(statement=s, reason=f"r{i}", verdict=v)
                for i, (s, v) in enumerate(self.verdicts)
            ])
        raise AssertionError(f"unexpected schema {schema!r}")


CONTEXT = [Chunk(chunk_id=f"{PASSAGES[0].passage_id}::0", source_passage_id=PASSAGES[0].passage_id,
                 text=PASSAGES[0].text[:120])]


def faith(judge, question="q?", answer="an answer", context=CONTEXT):
    return Faithfulness(llm=judge).score(question=question, answer=answer, context=context)


def test_faithfulness_is_supported_over_total_with_details():
    stmts = ["s1", "s2", "s3"]
    r = faith(FakeJudge(stmts, [("s1", 1), ("s2", 0), ("s3", 1)]))
    assert r.value == pytest.approx(2 / 3)
    assert r.details["statements"] == [
        {"statement": "s1", "verdict": True, "reason": "r0"},
        {"statement": "s2", "verdict": False, "reason": "r1"},
        {"statement": "s3", "verdict": True, "reason": "r2"},
    ]


def test_faithfulness_no_claims_is_nan_with_reason():
    judge = FakeJudge([], [])
    r = faith(judge)
    assert math.isnan(r.value) and r.reason == "No claims extracted."
    assert len(judge.prompts) == 1          # NLI step not called


def test_faithfulness_verdict_count_mismatch_is_nan_with_reason():
    r = faith(FakeJudge(["s1", "s2"], [("s1", 1)]))
    assert math.isnan(r.value)
    assert r.reason == "judge returned 1 verdicts for 2 statements"


@pytest.mark.parametrize("field", ["question", "answer", "context"])
def test_faithfulness_missing_input_raises(field):
    kwargs = {"question": "q?", "answer": "a", "context": CONTEXT, field: [] if field == "context" else ""}
    with pytest.raises(ValueError, match=f"{field} is missing"):
        Faithfulness(llm=FakeJudge(["s"], [("s", 1)])).score(**kwargs)


def test_faithfulness_judge_prompt_tags_context_with_passage_ids():
    judge = FakeJudge(["s1", "s2"], [("s1", 1), ("s2", 1)])
    faith(judge)
    nli_prompt = judge.prompts[1]
    assert f"[{PASSAGES[0].passage_id}]" in nli_prompt
    assert "1. s1\n2. s2" in nli_prompt


# ---------------------------------------------------------------- gate metadata (read by the comparator)

@pytest.mark.parametrize("cls,name,inputs", [
    (RecallAtK, "recall_at_k", {"retrieved_ids", "gold_ids"}),
    (MRR, "mrr", {"retrieved_ids", "gold_ids"}),
    (TokenPRF1, "token_prf1", {"answer", "gold_answers"}),
    (Faithfulness, "faithfulness", {"question", "answer", "context"}),
])
def test_metric_gate_metadata(cls, name, inputs):
    assert (cls.name, cls.required_inputs, cls.regression_threshold, cls.higher_is_better) == (
        name, frozenset(inputs), 0.05, True)
    assert METRIC_REGISTRY[name] is cls


def test_base_metric_defaults_higher_is_better():
    # D10-Q16: a real class attribute, not just an annotation
    assert Metric.higher_is_better is True
    assert Metric.regression_threshold is None