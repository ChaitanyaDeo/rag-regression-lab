"""
Day 12 B4.2 follow-up (F-D12-23): what _build_builtin_rag passes to BuiltinRAGSystem.

The factory must make the SAME constructor call as before the reranker existed
when the reranker is off, and add reranker kwargs only when it is configured.
Test fakes with a narrow __init__ (e.g. test_seed_wiring.py) depend on that.

No models, no Chroma: BuiltinRAGSystem / ChunkerConfig / CrossEncoderReranker are faked.
"""

import sys
import types

import pytest

from ragregress.rag.registry import BuiltinRAGParams, _build_builtin_rag

BASE = dict(embedding_model="all-MiniLM-L6-v2", chunk_size=500, overlap=50,
            generation_model="qwen2.5:3b-instruct")
RERANKER = {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "candidates": 20}


class FakeChunkerConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name


@pytest.fixture()
def stubs(monkeypatch):
    seen = {}

    class LooseSystem:                       # accepts anything, records the exact call
        def __init__(self, *args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs

    monkeypatch.setitem(sys.modules, "ragregress.rag.builtin_system",
                        types.SimpleNamespace(BuiltinRAGSystem=LooseSystem))
    monkeypatch.setitem(sys.modules, "ragregress.rag.chunker",
                        types.SimpleNamespace(ChunkerConfig=FakeChunkerConfig))
    monkeypatch.setitem(sys.modules, "ragregress.rag.reranker",
                        types.SimpleNamespace(CrossEncoderReranker=FakeCrossEncoder))
    return seen


def test_off_passes_no_reranker_kwargs(stubs):
    _build_builtin_rag(BuiltinRAGParams(**BASE))
    assert set(stubs["kwargs"]) == {"generator_config", "embedding_model"}
    assert len(stubs["args"]) == 1                         # the ChunkerConfig


def test_off_call_fits_the_pre_reranker_signature(monkeypatch):
    class NarrowSystem:                                    # the Day 10 constructor shape
        def __init__(self, chunker_config, generator_config=None, embedding_model=None):
            self.embedding_model = embedding_model

    monkeypatch.setitem(sys.modules, "ragregress.rag.builtin_system",
                        types.SimpleNamespace(BuiltinRAGSystem=NarrowSystem))
    monkeypatch.setitem(sys.modules, "ragregress.rag.chunker",
                        types.SimpleNamespace(ChunkerConfig=FakeChunkerConfig))
    assert _build_builtin_rag(BuiltinRAGParams(**BASE)).embedding_model == "all-MiniLM-L6-v2"


def test_on_passes_reranker_and_candidates(stubs):
    _build_builtin_rag(BuiltinRAGParams(**BASE, reranker=RERANKER))
    kw = stubs["kwargs"]
    assert set(kw) == {"generator_config", "embedding_model", "reranker", "rerank_candidates"}
    assert isinstance(kw["reranker"], FakeCrossEncoder)
    assert kw["reranker"].model_name == RERANKER["model"]
    assert kw["rerank_candidates"] == 20


def test_on_still_passes_seed_and_embedding_model(stubs):
    _build_builtin_rag(BuiltinRAGParams(**BASE, seed=11, reranker=RERANKER))
    assert stubs["kwargs"]["generator_config"].seed == 11
    assert stubs["kwargs"]["embedding_model"] == "all-MiniLM-L6-v2"


def test_off_does_not_touch_the_reranker_module(monkeypatch, stubs):
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("reranker constructed while off")

    monkeypatch.setitem(sys.modules, "ragregress.rag.reranker",
                        types.SimpleNamespace(CrossEncoderReranker=Boom))
    _build_builtin_rag(BuiltinRAGParams(**BASE))