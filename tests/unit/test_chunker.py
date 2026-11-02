"""
tests/unit/test_chunker.py

ragregress/rag/chunker.py fixed-size character chunking via
CharacterTextSplitter(separator="").

Pinned behaviour (D10-Q11, option A -- characterise, don't change):
  - On text with no whitespace the windows are exact: length chunk_size, step chunk_size - overlap.
  - The splitter strips whitespace at chunk edges (strip_whitespace=True default), so on real
    text a chunk can be shorter than chunk_size and the overlap is not exact (F-D10-13).
    Real-text tests therefore check "<= chunk_size and a substring of the passage" only.
D10-Q12: ChunkerConfig rejects overlap >= chunk_size itself (the splitter accepts ==).
D10-Q13: chunk counts on tests/fixtures/passages_5doc.json are pinned (verified on
langchain-text-splitters 1.1.2 and 1.1.3); a library change that alters chunking fails here.

Run (repo root):  python -m pytest tests/unit/test_chunker.py -v
"""

from __future__ import annotations

import json
import string
from dataclasses import asdict
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage
from ragregress.rag.chunker import ChunkerConfig, chunk_passage, chunk_passages

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


PASSAGES = [Passage(**p) for p in load_fixture("passages_5doc.json")]
LETTERS = string.ascii_letters  # 52 distinct chars, no whitespace


def passage(text: str, pid: str = "p0000001", title: str = "T") -> Passage:
    return Passage(passage_id=pid, title=title, text=text)


def texts(text: str, size: int, overlap: int):
    return [c.text for c in chunk_passage(passage(text), ChunkerConfig(chunk_size=size, overlap=overlap))]


def exact_windows(text: str, size: int, overlap: int):
    """Reference implementation for whitespace-free text."""
    step, out, start = size - overlap, [], 0
    while True:
        out.append(text[start:start + size])
        if start + size >= len(text):
            return out
        start += step


# ---------------------------------------------------------------- boundaries

def test_text_shorter_than_chunk_size_is_one_identical_chunk():
    assert texts("hello", 10, 2) == ["hello"]


def test_text_exactly_chunk_size_is_one_chunk():
    assert texts(LETTERS[:10], 10, 3) == [LETTERS[:10]]


def test_overlap_zero_gives_disjoint_full_windows():
    assert texts(LETTERS[:20], 10, 0) == [LETTERS[:10], LETTERS[10:20]]


def test_last_chunk_is_shorter():
    assert [len(t) for t in texts(LETTERS[:30], 10, 3)] == [10, 10, 10, 9]


def test_empty_text_gives_no_chunks():
    assert chunk_passage(passage(""), ChunkerConfig()) == []


# ---------------------------------------------------------------- overlap (whitespace-free text)

@pytest.mark.parametrize("size,overlap", [(10, 0), (10, 3), (7, 6), (52, 10), (5, 1)])
def test_windows_match_reference(size, overlap):
    assert texts(LETTERS, size, overlap) == exact_windows(LETTERS, size, overlap)


def test_consecutive_chunks_share_exactly_overlap_chars_and_rebuild_the_text():
    chunks = texts(LETTERS, 10, 3)
    assert all(nxt.startswith(prev[-3:]) for prev, nxt in zip(chunks, chunks[1:]))
    assert chunks[0] + "".join(c[3:] for c in chunks[1:]) == LETTERS


# ---------------------------------------------------------------- identity / metadata

def test_chunk_ids_source_and_metadata():
    chunks = chunk_passage(passage(LETTERS[:30], pid="p0000042", title="Some_Title"),
                           ChunkerConfig(chunk_size=10, overlap=3))
    assert [c.chunk_id for c in chunks] == [f"p0000042::{i}" for i in range(4)]
    assert {c.source_passage_id for c in chunks} == {"p0000042"}
    assert all(c.metadata == {"title": "Some_Title"} for c in chunks)


def test_chunk_passages_keeps_passage_order_and_ids_are_unique():
    chunks = chunk_passages(PASSAGES, ChunkerConfig(chunk_size=500, overlap=50))
    order = list(dict.fromkeys(c.source_passage_id for c in chunks))
    assert order == [p.passage_id for p in PASSAGES]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_is_deterministic():
    cfg = ChunkerConfig(chunk_size=500, overlap=50)
    assert chunk_passages(PASSAGES, cfg) == chunk_passages(PASSAGES, cfg)


# ---------------------------------------------------------------- real text (fixture)

CONFIGS = [(500, 50), (1000, 50)]


@pytest.mark.parametrize("size,overlap", CONFIGS)
def test_real_chunks_fit_and_come_from_their_passage(size, overlap):
    text = {p.passage_id: p.text for p in PASSAGES}
    for c in chunk_passages(PASSAGES, ChunkerConfig(chunk_size=size, overlap=overlap)):
        assert 0 < len(c.text) <= size
        assert c.text in text[c.source_passage_id]


@pytest.mark.parametrize("size,overlap", CONFIGS)
def test_real_chunks_have_no_edge_whitespace(size, overlap):
    # current behaviour: the splitter strips chunk edges (F-D10-13)
    chunks = chunk_passages(PASSAGES, ChunkerConfig(chunk_size=size, overlap=overlap))
    assert all(c.text == c.text.strip() for c in chunks)


@pytest.mark.parametrize("size,overlap,expected", [
    (500, 50, [2, 2, 3, 1, 2]),
    (1000, 50, [1, 1, 2, 1, 1]),
])
def test_fixture_chunk_counts_are_pinned(size, overlap, expected):
    cfg = ChunkerConfig(chunk_size=size, overlap=overlap)
    assert [len(chunk_passage(p, cfg)) for p in PASSAGES] == expected


# ---------------------------------------------------------------- validation (D10-Q12)

@pytest.mark.parametrize("size,overlap", [(5, 5), (5, 6), (500, 500)])
def test_overlap_not_less_than_chunk_size_is_rejected(size, overlap):
    with pytest.raises(ValueError, match="must be less than chunk_size"):
        ChunkerConfig(chunk_size=size, overlap=overlap)


def test_config_fields_unchanged_so_collection_fingerprint_is_stable():
    # retriever._chunker_figerprint hashes asdict(ChunkerConfig); validation must not add fields
    assert asdict(ChunkerConfig()) == {"chunk_size": 500, "overlap": 50}