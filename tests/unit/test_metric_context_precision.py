"""ContextPrecision: Ragas-style average precision over per-chunk verdicts, with a fake judge (no model)."""
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ragregress.evaluation.metrics.context_precision.metric import ContextPrecision
from ragregress.evaluation.metrics.context_precision.utils import ContextPrecisionVerdict


class FakeJudge:
    """Returns the scripted verdicts in call order and records the prompts."""

    def __init__(self, verdicts):
        self.verdicts, self.prompts, self.schemas = list(verdicts), [], []

    def generate_structured(self, prompt, schema):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return ContextPrecisionVerdict(reason=f"r{len(self.prompts)}", verdict=self.verdicts[len(self.prompts) - 1])


def chunks(n):
    return [SimpleNamespace(source_passage_id=f"p{i}", text=f"text {i}") for i in range(n)]


def precision(judge, n=None, reference="Roger Goodell.", question="Who?"):
    n = len(judge.verdicts) if n is None else n
    return ContextPrecision(llm=judge).score(question=question, reference=reference, context=chunks(n))


@pytest.mark.parametrize("verdicts,expected", [
    ([1, 1, 1], 1.0),
    ([0, 0, 0], 0.0),
    ([1, 0, 0], 1.0),                                  # useful chunk first
    ([0, 0, 1], 1 / 3),                                # same chunk last
    ([1, 0, 1], (1 / 1 + 2 / 3) / 2),
    ([0, 1, 1, 0], (1 / 2 + 2 / 3) / 2),
])
def test_average_precision(verdicts, expected):
    assert precision(FakeJudge(verdicts)).value == pytest.approx(expected)


def test_useful_chunks_ranked_first_beat_the_same_chunks_ranked_last():
    assert precision(FakeJudge([1, 1, 0, 0])).value > precision(FakeJudge([0, 0, 1, 1])).value


def test_one_call_per_chunk_with_the_verdict_schema():
    judge = FakeJudge([1, 0, 1, 0, 0])
    precision(judge)
    assert len(judge.prompts) == 5 and set(judge.schemas) == {ContextPrecisionVerdict}


def test_number_of_calls_follows_the_number_of_chunks():
    for n in (1, 3, 7):
        judge = FakeJudge([1] * n)
        precision(judge)
        assert len(judge.prompts) == n


def test_each_prompt_has_question_reference_and_only_its_own_chunk():
    judge = FakeJudge([1, 0])
    precision(judge, reference="Roger Goodell.\nGoodell.")
    first, second = judge.prompts
    assert "Who?" in first and "Roger Goodell.\nGoodell." in first
    assert "[p0]" in first and "text 0" in first and "[p1]" not in first
    assert "[p1]" in second and "text 1" in second and "[p0]" not in second


def test_details_list_each_chunk_with_verdict_and_reason():
    r = precision(FakeJudge([1, 0]))
    assert r.details["chunks"] == [
        {"passage_id": "p0", "verdict": True, "reason": "r1"},
        {"passage_id": "p1", "verdict": False, "reason": "r2"},
    ]


@pytest.mark.parametrize("empty", ["", "  "])
def test_empty_reference_is_nan_without_a_judge_call(empty):
    judge = FakeJudge([1])
    r = precision(judge, n=1, reference=empty)
    assert math.isnan(r.value) and "No-answer" in r.reason
    assert judge.prompts == []


def test_a_failing_judge_call_raises():
    class Boom:
        def generate_structured(self, prompt, schema):
            raise RuntimeError("judge down")

    with pytest.raises(RuntimeError):
        ContextPrecision(llm=Boom()).score(question="q", reference="a.", context=chunks(2))


@pytest.mark.parametrize("kwargs", [{"reference": None}, {"reference": ["a"]}, {"question": ""}])
def test_bad_input_raises(kwargs):
    with pytest.raises(ValueError):
        precision(FakeJudge([1]), n=1, **kwargs)


def test_empty_context_raises():
    with pytest.raises(ValueError):
        ContextPrecision(llm=FakeJudge([])).score(question="q", reference="a.", context=[])


@pytest.mark.parametrize("bad", [2, -1])
def test_verdict_outside_0_1_is_rejected(bad):
    with pytest.raises(ValidationError):
        ContextPrecisionVerdict(reason="r", verdict=bad)