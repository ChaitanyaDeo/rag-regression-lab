"""ContextRecall: Ragas-style attribution scoring with a fake judge (no model)."""
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ragregress.evaluation.metrics.context_recall.metric import ContextRecall
from ragregress.evaluation.metrics.context_recall.utils import ContextRecallClassification, ContextRecallOutput


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts, self.prompts, self.schemas = verdicts, [], []

    def generate_structured(self, prompt, schema):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return ContextRecallOutput(classifications=[
            ContextRecallClassification(statement=f"s{i}", reason="r", attributed=v)
            for i, v in enumerate(self.verdicts)
        ])


CONTEXT = [SimpleNamespace(source_passage_id="p1", text="Roger Goodell is the NFL commissioner."),
           SimpleNamespace(source_passage_id="p2", text="Second chunk text.")]


def recall(judge, reference="Roger Goodell.", context=CONTEXT, question="Who is the commissioner?"):
    return ContextRecall(llm=judge).score(question=question, reference=reference, context=context)


def test_score_is_attributed_over_total_with_details():
    r = recall(FakeJudge([1, 0, 1, 1]))
    assert r.value == pytest.approx(0.75)
    assert [s["verdict"] for s in r.details["statements"]] == [True, False, True, True]


def test_unsupported_reference_scores_zero():
    assert recall(FakeJudge([0])).value == 0.0


def test_one_call_with_the_output_schema():
    judge = FakeJudge([1, 1])
    recall(judge, reference="Roger Goodell.\nGoodell.")
    assert len(judge.prompts) == 1 and judge.schemas == [ContextRecallOutput]


def test_prompt_has_question_reference_as_given_and_every_chunk():
    judge = FakeJudge([1, 1])
    recall(judge, reference="Roger Goodell.\nGoodell.")
    prompt = judge.prompts[0]
    assert "Who is the commissioner?" in prompt
    assert prompt.split("Answer:")[-1].strip() == "Roger Goodell.\nGoodell."
    assert "[p1]" in prompt and "[p2]" in prompt and "Second chunk text." in prompt


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_reference_is_nan_without_a_judge_call(empty):
    judge = FakeJudge([1])
    r = recall(judge, reference=empty)
    assert math.isnan(r.value) and "No-answer" in r.reason
    assert judge.prompts == []


@pytest.mark.parametrize("kwargs", [{"reference": None}, {"reference": ["a"]}, {"context": []}, {"question": ""}])
def test_bad_input_raises(kwargs):
    with pytest.raises(ValueError):
        recall(FakeJudge([1]), **kwargs)


@pytest.mark.parametrize("bad", [2, -1])
def test_verdict_outside_0_1_is_rejected(bad):
    with pytest.raises(ValidationError):
        ContextRecallClassification(statement="s", reason="r", attributed=bad)


def test_empty_classification_list_is_rejected():
    with pytest.raises(ValidationError):
        ContextRecallOutput(classifications=[])