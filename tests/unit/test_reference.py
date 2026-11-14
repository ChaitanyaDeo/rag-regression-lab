"""build_reference: the gold answers as one reference text."""
import pytest

from ragregress.evaluation.reference import build_reference


def test_one_sentence_per_gold_in_order():
    assert build_reference(["Roger Goodell", "Goodell"]) == "Roger Goodell.\nGoodell."


def test_golds_equal_after_normalization_are_listed_once():
    assert build_reference(["Roger Goodell", "roger goodell", "Roger  Goodell!", "the Roger Goodell"]) == "Roger Goodell."


def test_existing_end_punctuation_is_kept():
    assert build_reference(["It rained?", "Yes!"]) == "It rained?\nYes!"


def test_empty_golds_are_dropped():
    assert build_reference(["", "Goodell", "  "]) == "Goodell."


@pytest.mark.parametrize("golds", [[""], [], None, ["", " "]])
def test_no_answer_gives_empty_text(golds):
    assert build_reference(golds) == ""