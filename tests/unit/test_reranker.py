"""
tests/unit/test_reranker.py

cross-encoder reranking inside BuiltinRAGSystem.

  A. CrossEncoderReranker with a FAKE cross-encoder (no model, no download)
  B. BuiltinRAGSystem.retrieve() hook: fake embedder + REAL Chroma in a temp folder
  C. registry factory: reranker block -> reranker passed into the system

Safety (F-D10-33): every Chroma instance lives in tmp_path; the factory tests chdir into
tmp_path because the factory uses the relative default artifacts/chroma. Nothing here
touches the real ragregress.db or artifacts/chroma, and no model is loaded.

The real cross-encoder is covered by an opt-in test (RAGREGRESS_SLOW=1), see the bottom.

Run (repo root):  python -m pytest tests/unit/test_reranker.py -v
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

import pytest

from ragregress.benchmark.schemas import Passage
from ragregress.rag import builtin_system
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.rag.interface import Chunk, RAGSystem
from ragregress.rag.registry import BuiltinRAGParams, build_rag_system
from ragregress.rag.reranker import CrossEncoderReranker, Reranker

SLOW = os.environ.get("RAGREGRESS_SLOW") == "1"
CFG = ChunkerConfig(chunk_size=500, overlap=50)
QUERY = "river bank erosion flood control"


# ---------------------------------------------------------------- fakes

class FakeCrossEncoder:
    """predict() scores a pair by a lookup on the chunk text (default 0.0); records calls."""

    def __init__(self, scores=None, default=0.0):
        self.scores = scores or {}
        self.default = default
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.scores.get(text, self.default) for _, text in pairs]


class KeywordCrossEncoder:
    """Scores 10 for a text containing GOLD, else 0 (stands in for 'understands relevance')."""

    def predict(self, pairs):
        return [10.0 if "GOLD" in text else 0.0 for _, text in pairs]


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims, L2-normalised; stands in for embedder.Embedder."""

    def __init__(self, model_name, device=None):
        self.model_name = model_name

    def encode(self, texts, **_):
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class SpyReranker:
    """Records what retrieve() hands over; returns the LAST k candidates (distinguishable order)."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, chunks, k):
        self.calls.append((query, [c.chunk_id for c in chunks], k))
        return list(reversed(chunks))[:k]


def chunks(*texts):
    return [Chunk(chunk_id=f"p{i}::0", source_passage_id=f"p{i}", text=t, metadata={"title": f"T{i}"})
            for i, t in enumerate(texts)]


def ids(cs):
    return [c.chunk_id for c in cs]


# Vector search ranks (fake embedder, QUERY): d3 d2 d0 d1 d4 GOLD d6 d5  -> gold is 6th
PASSAGES = [Passage(passage_id=k, title=k, text=v) for k, v in {
    "d0": "river bank erosion flood control measures along the river bank",
    "d1": "flood control and river erosion bank protection projects",
    "d2": "bank erosion control after river flood events",
    "d3": "river flood control bank erosion policy",
    "d4": "control of flood and erosion near the river bank",
    "gold": "the answer is GOLD marker unrelated words banana apple orchard harvest river",
    "d5": "completely different topic about mountain climbing and snow",
    "d6": "cooking pasta with tomato sauce and basil",
}.items()]
VECTOR_ORDER = ["d3", "d2", "d0", "d1", "d4", "gold", "d6", "d5"]


def make_system(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(builtin_system, "Embedder", FakeEmbedder)
    system = BuiltinRAGSystem(CFG, embedding_model="fake-model",
                              persist_directory=str(tmp_path / "chroma"), **kwargs)
    system.ensure_indexed(PASSAGES)
    return system


# ================================================================ A. CrossEncoderReranker

def test_reorders_by_score_descending_and_keeps_k():
    cs = chunks("a", "b", "c", "d")
    r = CrossEncoderReranker("m", model=FakeCrossEncoder({"a": 1.0, "b": 4.0, "c": 3.0, "d": 2.0}))
    assert ids(r.rerank("q", cs, 3)) == ["p1::0", "p2::0", "p3::0"]


def test_k_larger_than_candidates_returns_all_reranked():
    cs = chunks("a", "b")
    r = CrossEncoderReranker("m", model=FakeCrossEncoder({"a": 1.0, "b": 2.0}))
    assert ids(r.rerank("q", cs, 10)) == ["p1::0", "p0::0"]


def test_ties_keep_the_incoming_order():
    cs = chunks("a", "b", "c", "d")
    r = CrossEncoderReranker("m", model=FakeCrossEncoder({"c": 5.0}, default=1.0))
    assert ids(r.rerank("q", cs, 4)) == ["p2::0", "p0::0", "p1::0", "p3::0"]


def test_nan_scores_rank_last():
    cs = chunks("a", "b", "c")
    r = CrossEncoderReranker("m", model=FakeCrossEncoder({"a": float("nan"), "b": -3.0, "c": -9.0}))
    assert ids(r.rerank("q", cs, 3)) == ["p1::0", "p2::0", "p0::0"]


def test_negative_logits_are_ordered_correctly():
    cs = chunks("a", "b")
    r = CrossEncoderReranker("m", model=FakeCrossEncoder({"a": -11.2, "b": -0.5}))
    assert ids(r.rerank("q", cs, 1)) == ["p1::0"]


def test_model_receives_query_text_pairs_in_incoming_order():
    cs = chunks("first", "second", "third")
    fake = FakeCrossEncoder()
    CrossEncoderReranker("m", model=fake).rerank("my query", cs, 2)
    assert fake.calls == [[("my query", "first"), ("my query", "second"), ("my query", "third")]]


def test_chunks_are_returned_unmodified():
    cs = chunks("a", "b")
    before = [c.model_copy(deep=True) for c in cs]
    out = CrossEncoderReranker("m", model=FakeCrossEncoder({"b": 1.0})).rerank("q", cs, 2)
    assert out[0] is cs[1] and out[1] is cs[0]                     # same objects
    assert [c.model_dump() for c in cs] == [c.model_dump() for c in before]


def test_empty_candidates_returns_empty_without_calling_the_model():
    fake = FakeCrossEncoder()
    assert CrossEncoderReranker("m", model=fake).rerank("q", [], 5) == []
    assert fake.calls == []


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_rejected(k):
    with pytest.raises(ValueError):
        CrossEncoderReranker("m", model=FakeCrossEncoder()).rerank("q", chunks("a"), k)


def test_score_count_mismatch_raises():
    class Short:
        def predict(self, pairs):
            return [1.0]

    with pytest.raises(RuntimeError, match="1 scores for 2 chunks"):
        CrossEncoderReranker("m", model=Short()).rerank("q", chunks("a", "b"), 1)


def test_numpy_like_scores_are_accepted():
    class ArrayLike:
        def predict(self, pairs):
            import array
            return array.array("f", [0.5, 2.5, 1.5])

    out = CrossEncoderReranker("m", model=ArrayLike()).rerank("q", chunks("a", "b", "c"), 2)
    assert ids(out) == ["p1::0", "p2::0"]


def test_injected_model_means_no_real_model_is_loaded(monkeypatch):
    import sentence_transformers

    class Bomb:
        def __init__(self, *a, **k):
            raise AssertionError("the real CrossEncoder must not load when a model is injected")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Bomb, raising=False)
    r = CrossEncoderReranker("some/model", model=FakeCrossEncoder())
    assert r.model_name == "some/model"


def test_without_a_model_the_named_one_is_loaded_lazily(monkeypatch):
    import sentence_transformers

    seen = {}

    class Spy:
        def __init__(self, name, device=None):
            seen.update(name=name, device=device)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Spy, raising=False)
    r = CrossEncoderReranker("cross-encoder/x", device="cpu")
    assert seen == {"name": "cross-encoder/x", "device": "cpu"}
    assert isinstance(r.model, Spy)


def test_reranker_satisfies_the_protocol():
    assert isinstance(CrossEncoderReranker("m", model=FakeCrossEncoder()), Reranker)


# ================================================================ B. BuiltinRAGSystem.retrieve hook

def test_precondition_gold_is_outside_the_vector_top_3_but_inside_the_top_6(tmp_path, monkeypatch):
    system = make_system(tmp_path, monkeypatch)
    order = [c.source_passage_id for c in system.retrieve(QUERY, k=8)]
    assert order == VECTOR_ORDER
    assert order.index("gold") == 5


def test_without_reranker_retrieve_is_plain_vector_search(tmp_path, monkeypatch):
    system = make_system(tmp_path, monkeypatch)
    assert system.reranker is None
    assert [c.source_passage_id for c in system.retrieve(QUERY, k=3)] == VECTOR_ORDER[:3]


def test_reranker_fetches_max_of_candidates_and_k_and_returns_its_choice(tmp_path, monkeypatch):
    spy = SpyReranker()
    system = make_system(tmp_path, monkeypatch, reranker=spy, rerank_candidates=6)
    out = system.retrieve(QUERY, k=3)

    (query, seen_ids, k), = spy.calls
    assert (query, k) == (QUERY, 3)
    assert [i.split("::")[0] for i in seen_ids] == VECTOR_ORDER[:6]      # 6 candidates, vector order
    assert [c.source_passage_id for c in out] == ["gold", "d4", "d1"]    # = last 3 of the 6, reversed


def test_k_larger_than_candidates_still_fetches_k(tmp_path, monkeypatch):
    spy = SpyReranker()
    system = make_system(tmp_path, monkeypatch, reranker=spy, rerank_candidates=2)
    system.retrieve(QUERY, k=5)
    (_, seen_ids, k), = spy.calls
    assert (len(seen_ids), k) == (5, 5)


def test_reranking_pulls_a_low_ranked_gold_chunk_to_rank_one(tmp_path, monkeypatch):
    reranker = CrossEncoderReranker("m", model=KeywordCrossEncoder())
    system = make_system(tmp_path, monkeypatch, reranker=reranker, rerank_candidates=6)
    top = system.retrieve(QUERY, k=3)
    assert top[0].source_passage_id == "gold"
    assert [c.source_passage_id for c in top[1:]] == VECTOR_ORDER[:2]     # ties keep vector order


def test_candidates_too_few_means_gold_is_not_found(tmp_path, monkeypatch):
    reranker = CrossEncoderReranker("m", model=KeywordCrossEncoder())
    system = make_system(tmp_path, monkeypatch, reranker=reranker, rerank_candidates=5)
    assert "gold" not in [c.source_passage_id for c in system.retrieve(QUERY, k=3)]


def test_health_check_probe_with_k_1_works(tmp_path, monkeypatch):
    # Pipeline._ingest calls retrieve("health check probe", k=1)
    reranker = CrossEncoderReranker("m", model=KeywordCrossEncoder())
    system = make_system(tmp_path, monkeypatch, reranker=reranker, rerank_candidates=4)
    assert len(system.retrieve("health check probe", k=1)) == 1


def test_reranker_does_not_change_the_index(tmp_path, monkeypatch):
    plain = make_system(tmp_path, monkeypatch)
    with_rr = make_system(tmp_path, monkeypatch, reranker=SpyReranker(), rerank_candidates=4)
    assert with_rr.retriever.collection_name == plain.retriever.collection_name
    assert with_rr.ensure_indexed(PASSAGES) == 0                         # index already current


@pytest.mark.parametrize("candidates", [0, -3])
def test_reranker_without_positive_candidates_is_rejected(tmp_path, monkeypatch, candidates):
    monkeypatch.setattr(builtin_system, "Embedder", FakeEmbedder)
    with pytest.raises(ValueError, match="rerank_candidates"):
        BuiltinRAGSystem(CFG, embedding_model="fake", persist_directory=str(tmp_path / "c"),
                         reranker=SpyReranker(), rerank_candidates=candidates)


def test_candidates_without_reranker_are_ignored(tmp_path, monkeypatch):
    system = make_system(tmp_path, monkeypatch, rerank_candidates=6)
    assert [c.source_passage_id for c in system.retrieve(QUERY, k=3)] == VECTOR_ORDER[:3]


def test_system_with_reranker_still_satisfies_the_ragsystem_protocol(tmp_path, monkeypatch):
    assert isinstance(make_system(tmp_path, monkeypatch, reranker=SpyReranker(), rerank_candidates=4), RAGSystem)


# ================================================================ C. registry factory

PARAMS = dict(embedding_model="fake-model", chunk_size=500, overlap=50,
              generation_model="qwen2.5:3b-instruct", seed=42)


@pytest.fixture()
def factory_env(tmp_path, monkeypatch):
    """Fake embedder + cwd in tmp_path (the factory uses the relative default artifacts/chroma)."""
    monkeypatch.setattr(builtin_system, "Embedder", FakeEmbedder)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_factory_without_reranker_builds_a_plain_system(factory_env):
    system = build_rag_system("builtin_rag", BuiltinRAGParams(**PARAMS))
    assert system.reranker is None and system.rerank_candidates == 0
    assert (factory_env / "artifacts" / "chroma").exists()


def test_factory_with_reranker_builds_the_cross_encoder_and_passes_candidates(factory_env, monkeypatch):
    import ragregress.rag.reranker as reranker_module

    built = []

    class SpyCrossEncoderReranker:
        def __init__(self, model_name, *a, **k):
            built.append(model_name)

        def rerank(self, query, chunks, k):
            return chunks[:k]

    monkeypatch.setattr(reranker_module, "CrossEncoderReranker", SpyCrossEncoderReranker)
    params = BuiltinRAGParams(**PARAMS, reranker={"model": "cross-encoder/x", "candidates": 20})
    system = build_rag_system("builtin_rag", params)

    assert built == ["cross-encoder/x"]
    assert isinstance(system.reranker, SpyCrossEncoderReranker)
    assert system.rerank_candidates == 20


# ================================================================ opt-in: the real cross-encoder (D's machine)

@pytest.mark.skipif(not SLOW, reason="loads the real cross-encoder; set RAGREGRESS_SLOW=1")
def test_real_cross_encoder_ranks_the_relevant_passage_first():
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cs = [
        Chunk(chunk_id="a::0", source_passage_id="a", text="Paris is the capital of France."),
        Chunk(chunk_id="b::0", source_passage_id="b", text="Hamlet was written by William Shakespeare."),
        Chunk(chunk_id="c::0", source_passage_id="c", text="The Rhine flows through Germany."),
    ]
    out = reranker.rerank("who wrote hamlet", cs, 2)
    assert [c.chunk_id for c in out][0] == "b::0"
    assert len(out) == 2