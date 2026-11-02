"""
tests/unit/test_fixtures.py

Self-check for the shared test fixtures in tests/fixtures/ (D10-B2).
A broken fixture fails here, not as a confusing failure in the retriever / pipeline tests.

Run (repo root):  python -m pytest tests/unit/test_fixtures.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage, QAPair

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


PASSAGES = [Passage(**p) for p in load_fixture("passages_5doc.json")]
QA = [QAPair(**q) for q in load_fixture("qa_5doc.json")]
TEXT = {p.passage_id: p.text for p in PASSAGES}


def test_sizes():
    assert (len(PASSAGES), len(QA)) == (5, 5)


def test_ids_are_unique():
    assert len({p.passage_id for p in PASSAGES}) == 5
    assert len({q.qa_id for q in QA}) == 5


def test_qa_order_matches_passage_order():
    # qa[i] is about passages[i]; the 3-doc fixture is the first 3 of each
    assert [q.gold_passage_ids for q in QA] == [[p.passage_id] for p in PASSAGES]


def test_titles_match_their_gold_passage():
    title = {p.passage_id: p.title for p in PASSAGES}
    assert all(q.title == title[q.gold_passage_ids[0]] for q in QA)


@pytest.mark.parametrize("qa", QA, ids=lambda q: q.qa_id[:8])
def test_every_answer_sits_at_its_answer_start(qa):
    text = TEXT[qa.gold_passage_ids[0]]
    assert len(qa.answer) == len(qa.answer_start)
    for answer, start in zip(qa.answer, qa.answer_start):
        assert text[start:start + len(answer)] == answer


def test_passage_ids_match_citation_format():
    # generator.CITATION_REGEX only parses ids of the form p<digits>
    assert all(p.passage_id[0] == "p" and p.passage_id[1:].isdigit() for p in PASSAGES)


def test_lengths_cover_single_and_multi_chunk_cases():
    # chunk_size 500 in baseline.yaml: need passages both shorter and longer than one chunk
    lengths = [len(p.text) for p in PASSAGES]
    assert min(lengths) < 500 < max(lengths)