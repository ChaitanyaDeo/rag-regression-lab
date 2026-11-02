"""
tests/unit/test_retriever.py

ragregress/rag/retriever.py (Chroma wrapper) and BuiltinRAGSystem.ensure_indexed (D10-B5).

  - Wrapper / indexing tests use a FAKE deterministic embedder (word-hash vectors, D10-Q18):
    fast, offline, exercise Chroma for real (temp folder per test).
  - One group uses the REAL MiniLM + Chroma on the 5-doc fixture (D10-Q19, on by default):
    needs the sentence-transformers model in the local Hugging Face cache.
  - ensure_indexed must record the content fingerprint so a second call is a no-op
    (D10-Q17 / F-D10-7: it used to upsert every time and never store the fingerprint).

Run (repo root):  python -m pytest tests/unit/test_retriever.py -v
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage, QAPair
from ragregress.rag import builtin_system
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig, chunk_passages
from ragregress.rag.interface import Chunk
from ragregress.rag.retriever import ChromaRetriever, collection_name, content_fingerprint

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


PASSAGES = [Passage(**p) for p in load_fixture("passages_5doc.json")]
QA = [QAPair(**q) for q in load_fixture("qa_5doc.json")]
CFG = ChunkerConfig(chunk_size=500, overlap=50)
REAL_MODEL = "all-MiniLM-L6-v2"   # as in configs/baseline.yaml


# ---------------------------------------------------------------- fake embedder

class FakeEmbeddings:
    """Bag-of-words hashed into 64 dims, L2-normalised. Deterministic, no model."""
    DIM = 64

    def _vec(self, text):
        v = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeEmbedder:
    """Stands in for embedder.Embedder inside BuiltinRAGSystem (no model load)."""
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self._e = FakeEmbeddings()

    def encode(self, texts, **_):
        return self._e.embed_documents(texts)


def retriever(tmp_path, name="coll"):
    return ChromaRetriever(FakeEmbeddings(), persist_directory=str(tmp_path / "chroma"), collection_name=name)


def chunk(cid, pid, text, title="T"):
    return Chunk(chunk_id=cid, source_passage_id=pid, text=text, metadata={"title": title})


@pytest.fixture()
def fake_system(tmp_path, monkeypatch):
    monkeypatch.setattr(builtin_system, "Embedder", FakeEmbedder)
    return BuiltinRAGSystem(CFG, embedding_model="fake-model", persist_directory=str(tmp_path / "chroma"))


# ---------------------------------------------------------------- naming / fingerprints (no Chroma)

def test_collection_name_is_stable_for_the_same_config():
    assert collection_name(REAL_MODEL, CFG) == collection_name(REAL_MODEL, ChunkerConfig(500, 50))


def test_collection_name_differs_when_only_overlap_differs():
    a, b = collection_name(REAL_MODEL, CFG), collection_name(REAL_MODEL, ChunkerConfig(500, 60))
    assert a != b and a.startswith("passages_all-MiniLM-L6-v2_500_") and b.startswith("passages_all-MiniLM-L6-v2_500_")


def test_collection_name_makes_model_names_safe():
    name = collection_name("sentence-transformers/all-MiniLM-L6-v2", CFG)
    assert name.startswith("passages_all-MiniLM-L6-v2_500_")
    assert collection_name("bge:small", CFG).startswith("passages_bge_small_500_")


def test_content_fingerprint_ignores_order_and_tracks_text():
    a, b = chunk("p1::0", "p1", "alpha"), chunk("p2::0", "p2", "beta")
    assert content_fingerprint([a, b]) == content_fingerprint([b, a])
    assert content_fingerprint([a, b]) != content_fingerprint([a, chunk("p2::0", "p2", "beta!")])


# ---------------------------------------------------------------- ChromaRetriever (fake embeddings, real Chroma)

def test_upsert_then_count(tmp_path):
    r = retriever(tmp_path)
    r.upsert([chunk("p1::0", "p1", "steam engine boiler"), chunk("p2::0", "p2", "lake constance rhine")])
    assert r.count() == 2


def test_upsert_nothing_is_a_no_op(tmp_path):
    r = retriever(tmp_path)
    r.upsert([])
    assert r.count() == 0


def test_upsert_same_ids_twice_does_not_duplicate(tmp_path):
    r = retriever(tmp_path)
    chunks = [chunk("p1::0", "p1", "steam engine boiler")]
    r.upsert(chunks)
    r.upsert(chunks)
    assert r.count() == 1


def test_query_returns_chunks_with_ids_and_metadata_round_tripped(tmp_path):
    r = retriever(tmp_path)
    r.upsert([chunk("p1::0", "p1", "steam engine boiler pressure", title="Steam_engine"),
              chunk("p2::0", "p2", "lake constance rhine obersee", title="Rhine")])
    top = r.query("rhine obersee lake", k=1)
    assert len(top) == 1
    assert (top[0].chunk_id, top[0].source_passage_id, top[0].metadata) == ("p2::0", "p2", {"title": "Rhine"})
    assert top[0].text == "lake constance rhine obersee"


def test_query_respects_k(tmp_path):
    r = retriever(tmp_path)
    r.upsert([chunk(f"p{i}::0", f"p{i}", f"text number {i}") for i in range(6)])
    assert len(r.query("text number", k=3)) == 3


def test_fresh_collection_has_no_fingerprint(tmp_path):
    assert retriever(tmp_path).stored_fingerprint() is None


def test_replace_all_drops_stale_chunks_and_records_fingerprint(tmp_path):
    r = retriever(tmp_path)
    r.upsert([chunk("old::0", "old", "stale text"), chunk("p1::0", "p1", "kept text")])
    new = [chunk("p1::0", "p1", "kept text"), chunk("p2::0", "p2", "new text")]
    r.replace_all(new, "fp-123")
    assert r.count() == 2
    assert {c.source_passage_id for c in r.query("stale text kept new", k=5)} == {"p1", "p2"}
    assert r.stored_fingerprint() == "fp-123"


def test_count_and_fingerprint_survive_a_new_instance(tmp_path):
    retriever(tmp_path).replace_all([chunk("p1::0", "p1", "persisted text")], "fp-abc")
    again = retriever(tmp_path)
    assert (again.count(), again.stored_fingerprint()) == (1, "fp-abc")


# ---------------------------------------------------------------- ensure_indexed (D10-Q17)

def test_ensure_indexed_first_call_embeds_all_chunks(fake_system):
    expected = len(chunk_passages(PASSAGES, CFG))
    assert fake_system.ensure_indexed(PASSAGES) == expected == 10


def test_ensure_indexed_second_call_is_a_no_op(fake_system):
    fake_system.ensure_indexed(PASSAGES)
    assert fake_system.ensure_indexed(PASSAGES) == 0
    assert fake_system.retriever.stored_fingerprint() == content_fingerprint(chunk_passages(PASSAGES, CFG))


def test_ensure_indexed_force_rebuilds(fake_system):
    fake_system.ensure_indexed(PASSAGES)
    assert fake_system.ensure_indexed(PASSAGES, force=True) == 10


def test_ensure_indexed_rebuilds_without_stale_chunks_when_corpus_shrinks(fake_system):
    fake_system.ensure_indexed(PASSAGES)
    smaller = PASSAGES[:3]
    expected = len(chunk_passages(smaller, CFG))
    assert fake_system.ensure_indexed(smaller) == expected
    assert fake_system.retriever.count() == expected
    gone = {p.passage_id for p in PASSAGES[3:]}
    hits = fake_system.retrieve("Super Bowl NFL Goodell chloroplast DNA plastome", k=expected)
    assert not gone & {c.source_passage_id for c in hits}


# ---------------------------------------------------------------- real MiniLM + Chroma (D10-Q19)

@pytest.fixture(scope="module")
def real_system(tmp_path_factory):
    system = BuiltinRAGSystem(CFG, embedding_model=REAL_MODEL,
                              persist_directory=str(tmp_path_factory.mktemp("chroma_real")))
    system.ensure_indexed(PASSAGES)
    return system


@pytest.mark.parametrize("qa", QA, ids=lambda q: q.title)
def test_real_retrieval_ranks_gold_passage_first(real_system, qa):
    top = real_system.retrieve(qa.question, k=5)
    assert top[0].source_passage_id == qa.gold_passage_ids[0], [c.source_passage_id for c in top]


def test_real_retrieval_returns_real_passage_ids(real_system):
    known = {p.passage_id for p in PASSAGES}
    for qa in QA:
        assert {c.source_passage_id for c in real_system.retrieve(qa.question, k=5)} <= known