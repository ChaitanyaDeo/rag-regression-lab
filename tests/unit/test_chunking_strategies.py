"""
tests/unit/test_chunking_strategies.py

The chunking strategies in ragregress/rag/chunker.py: fixed (the default, covered in test_chunker.py),
recursive, sentence, sliding_window.

Properties checked for EVERY strategy in STRATEGIES: chunks are exact substrings of the passage, within
chunk_size, lose no text, and come out the same every time. Strategy-specific tests pin boundaries on small
texts. Chunk counts on real passages are not pinned for the new strategies (they follow the splitter library).

Run (repo root):  python -m pytest tests/unit/test_chunking_strategies.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage
from ragregress.rag.chunker import STRATEGIES, ChunkerConfig, chunk_passage

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PASSAGES = [Passage(**p) for p in json.loads((FIXTURES / "passages_5doc.json").read_text(encoding="utf-8"))]

# One valid config per strategy. test_every_strategy_has_an_example fails when a strategy is added without one.
EXAMPLES = {
    "fixed": dict(chunk_size=300, overlap=30),
    "recursive": dict(chunk_size=300, overlap=30),
    "sentence": dict(chunk_size=300, overlap=60),
    "sliding_window": dict(chunk_size=300, overlap=0, options={"window": 3, "stride": 2}),
}
PROSE = " ".join(f"Sentence number {i} talks about thing {i} in some detail." for i in range(40))
TEXTS = {"prose": PROSE, **{p.passage_id: p.text for p in PASSAGES}}
NEW_STRATEGIES = ["recursive", "sentence", "sliding_window"]
SENTENCE_STRATEGIES = ["sentence", "sliding_window"]
END_MARKS = ".!?\"')"


def config(strategy: str, **override) -> ChunkerConfig:
    return ChunkerConfig(strategy=strategy, **{**EXAMPLES[strategy], **override})


def pieces(text: str, strategy: str, **override):
    return [c.text for c in chunk_passage(Passage(passage_id="p1", title="T", text=text), config(strategy, **override))]


def locate(text: str, chunks):
    """(start, end) of each chunk in the text; chunks come in text order, so search forward."""
    spans, cursor = [], 0
    for chunk in chunks:
        start = text.index(chunk, cursor)
        spans.append((start, start + len(chunk)))
        cursor = start + 1
    return spans


ALL = [(s, name) for s in EXAMPLES for name in TEXTS]
ALL_IDS = [f"{s}-{name}" for s, name in ALL]


def test_every_strategy_has_an_example():
    assert set(STRATEGIES) <= set(EXAMPLES)


# ---------------------------------------------------------------- properties of every strategy

@pytest.mark.parametrize("strategy,name", ALL, ids=ALL_IDS)
def test_chunks_are_substrings_within_chunk_size(strategy, name):
    text = TEXTS[name]
    for chunk in pieces(text, strategy):
        assert 0 < len(chunk) <= EXAMPLES[strategy]["chunk_size"]
        assert chunk in text


@pytest.mark.parametrize("strategy,name", ALL, ids=ALL_IDS)
def test_no_text_is_lost(strategy, name):
    text = TEXTS[name]
    covered = set()
    for start, end in locate(text, pieces(text, strategy)):
        covered.update(range(start, end))
    assert [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered] == []


@pytest.mark.parametrize("strategy", list(EXAMPLES))
def test_same_input_same_chunks(strategy):
    assert pieces(PROSE, strategy) == pieces(PROSE, strategy)


@pytest.mark.parametrize("strategy", list(EXAMPLES))
@pytest.mark.parametrize("text", ["", "   \n  "])
def test_empty_text_gives_no_chunks(strategy, text):
    assert pieces(text, strategy) == []


@pytest.mark.parametrize("strategy", list(EXAMPLES))
def test_chunk_ids_follow_the_chunk_order(strategy):
    chunks = chunk_passage(Passage(passage_id="p0000042", title="T", text=PROSE), config(strategy))
    assert [c.chunk_id for c in chunks] == [f"p0000042::{i}" for i in range(len(chunks))]
    assert {c.source_passage_id for c in chunks} == {"p0000042"}


# ---------------------------------------------------------------- boundaries

@pytest.mark.parametrize("strategy,name", [(s, n) for s in NEW_STRATEGIES for n in TEXTS],
                         ids=[f"{s}-{n}" for s in NEW_STRATEGIES for n in TEXTS])
def test_chunks_start_and_end_at_word_boundaries(strategy, name):
    text = TEXTS[name]
    for start, end in locate(text, pieces(text, strategy)):
        assert start == 0 or text[start - 1].isspace()
        assert end == len(text) or text[end].isspace()


@pytest.mark.parametrize("strategy,name", [(s, n) for s in SENTENCE_STRATEGIES for n in TEXTS],
                         ids=[f"{s}-{n}" for s in SENTENCE_STRATEGIES for n in TEXTS])
def test_chunks_start_and_end_at_sentence_boundaries(strategy, name):
    # chunk_size 500 is above the longest sentence here; a longer one is cut at a space (tests below)
    text = TEXTS[name]
    for start, end in locate(text, pieces(text, strategy, chunk_size=500)):
        assert text[:start].rstrip() == "" or text[:start].rstrip()[-1] in END_MARKS
        assert text[start:end][-1] in END_MARKS


def test_the_fixed_strategy_is_the_one_that_cuts_mid_word():
    # why the other strategies exist: this passage makes `fixed` start a chunk inside a word
    text = TEXTS["p0001023"]
    starts = [s for s, _ in locate(text, pieces(text, "fixed"))]
    assert any(s > 0 and not text[s - 1].isspace() for s in starts)


# ---------------------------------------------------------------- sentence

FOUR = "Aaa bbb. Ccc ddd. Eee fff. Ggg hhh."          # four sentences of 8 characters


def test_sentence_packs_whole_sentences():
    assert pieces(FOUR, "sentence", chunk_size=20, overlap=0) == ["Aaa bbb. Ccc ddd.", "Eee fff. Ggg hhh."]


def test_sentence_overlap_repeats_whole_sentences_that_fit():
    assert pieces(FOUR, "sentence", chunk_size=20, overlap=9) == [
        "Aaa bbb. Ccc ddd.", "Ccc ddd. Eee fff.", "Eee fff. Ggg hhh."]


def test_sentence_overlap_smaller_than_a_sentence_repeats_nothing():
    assert pieces(FOUR, "sentence", chunk_size=20, overlap=7) == ["Aaa bbb. Ccc ddd.", "Eee fff. Ggg hhh."]


def test_sentence_overlap_never_pushes_a_chunk_over_chunk_size():
    # carrying "Ccc ddd." (8 characters, within overlap 9) would make the second chunk 29 > 28 characters
    text = "Aaa bbb. Ccc ddd. Eee fff ggg hhh iii."
    chunks = pieces(text, "sentence", chunk_size=28, overlap=9)
    assert chunks == ["Aaa bbb. Ccc ddd.", "Eee fff ggg hhh iii."]


def test_sentence_keeps_a_paragraph_break_inside_the_chunk():
    text = "First part.\n\nSecond part."
    assert pieces(text, "sentence", chunk_size=100, overlap=0) == [text]


def test_a_sentence_longer_than_chunk_size_is_cut_at_spaces():
    text = " ".join(["word"] * 30) + "."
    chunks = pieces(text, "sentence", chunk_size=40, overlap=0)
    assert len(chunks) > 1 and all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks) == text


def test_a_single_word_longer_than_chunk_size_is_cut_anyway():
    word = "x" * 100
    chunks = pieces(word, "sentence", chunk_size=30, overlap=0)
    assert all(len(c) <= 30 for c in chunks) and "".join(chunks) == word


# ---------------------------------------------------------------- sliding_window

FIVE = "Aaa bbb. Ccc ddd. Eee fff. Ggg hhh. Iii jjj."


def window(text, w, s, size=500):
    return pieces(text, "sliding_window", chunk_size=size, options={"window": w, "stride": s})


def test_sliding_window_moves_by_the_stride():
    assert window(FIVE, 3, 1) == [
        "Aaa bbb. Ccc ddd. Eee fff.", "Ccc ddd. Eee fff. Ggg hhh.", "Eee fff. Ggg hhh. Iii jjj."]
    assert window(FIVE, 3, 2) == ["Aaa bbb. Ccc ddd. Eee fff.", "Eee fff. Ggg hhh. Iii jjj."]


def test_sliding_window_with_stride_equal_to_window_does_not_overlap():
    assert window(FIVE, 3, 3) == ["Aaa bbb. Ccc ddd. Eee fff.", "Ggg hhh. Iii jjj."]


def test_sliding_window_larger_than_the_text_is_one_chunk():
    assert window(FIVE, 9, 1) == [FIVE]


def test_sliding_window_of_one_sentence():
    assert window(FIVE, 1, 1) == ["Aaa bbb.", "Ccc ddd.", "Eee fff.", "Ggg hhh.", "Iii jjj."]


def test_sliding_window_shrinks_to_chunk_size_and_still_skips_no_sentence():
    # 3 sentences are 26 characters: over the limit, so each window keeps 2 and the next one starts after them
    assert window(FIVE, 3, 3, size=20) == ["Aaa bbb. Ccc ddd.", "Eee fff. Ggg hhh.", "Iii jjj."]


def test_sliding_window_keeps_one_sentence_even_when_it_exceeds_chunk_size():
    chunks = window("Short one. " + " ".join(["long"] * 20) + ".", 2, 1, size=30)
    assert all(len(c) <= 30 for c in chunks)


# ---------------------------------------------------------------- ChunkerConfig validation

def test_default_config_is_the_fixed_strategy():
    cfg = ChunkerConfig()
    assert (cfg.strategy, cfg.options) == ("fixed", {})


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown chunking strategy 'semantic'"):
        ChunkerConfig(strategy="semantic")


@pytest.mark.parametrize("strategy,options", [
    ("recursive", {"window": 2}),              # option the strategy does not have
    ("sliding_window", {"window": 3}),          # stride missing
    ("sliding_window", {}),
])
def test_options_must_match_the_strategy(strategy, options):
    with pytest.raises(ValueError, match="takes options"):
        ChunkerConfig(strategy=strategy, overlap=0, options=options)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3"])
def test_option_values_must_be_positive_integers(bad):
    with pytest.raises(ValueError, match="positive integer"):
        ChunkerConfig(strategy="sliding_window", overlap=0, options={"window": bad, "stride": 1})


def test_sliding_window_requires_zero_overlap():
    with pytest.raises(ValueError, match="does not use overlap"):
        ChunkerConfig(strategy="sliding_window", overlap=50, options={"window": 3, "stride": 1})


def test_stride_larger_than_window_is_rejected():
    with pytest.raises(ValueError, match="stride"):
        ChunkerConfig(strategy="sliding_window", overlap=0, options={"window": 2, "stride": 3})