"""
tests/unit/test_chunking_config.py

`pipeline.params.chunking`: how a chunking strategy is chosen in a run config, how it reaches
ChunkerConfig, and how it names the Chroma collection.

Guards:
  - chunking absent == `chunking: {strategy: fixed}`: same serialized params, same config_hash, same
    collection name as before strategies existed (no re-index, no new run identity)
  - other strategies and their options are hashed; wrong strategy / option / overlap is rejected at load
  - the pydantic models in registry.py match ragregress.rag.chunker.STRATEGIES
  - configs/chunk_*.yaml change only the chunking (and overlap where the strategy needs 0)

No models, no Chroma client, no DB.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Literal, get_args

import pytest
from pydantic import ValidationError

from ragregress.core.config import RunConfig, load_run_config
from ragregress.rag.chunker import STRATEGIES, ChunkerConfig
from ragregress.rag.registry import BuiltinRAGParams, ChunkingParams, _build_builtin_rag, chunker_kwargs
from ragregress.rag.retriever import _chunker_figerprint, collection_name

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
CHROMA_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")

BASE = dict(embedding_model="all-MiniLM-L6-v2", chunk_size=500, overlap=50, generation_model="qwen2.5:3b-instruct")
SLIDING = dict(strategy="sliding_window", window=3, stride=1)
OTHERS = {
    "recursive": ({"strategy": "recursive"}, 50),
    "sentence": ({"strategy": "sentence"}, 50),
    "sliding_window": (SLIDING, 0),
}
# The default strategy's collection fingerprint. Existing Chroma collections are named by it: changing it
# makes every baseline index be rebuilt. Update only on purpose.
DEFAULT_FINGERPRINT = "89ebe4585c6ae8c6"


def params(chunking=None, **override) -> BuiltinRAGParams:
    values = {**BASE, **override}
    if chunking is not None:
        values["chunking"] = chunking
    return BuiltinRAGParams(**values)


def run_config(chunking=None, **override) -> RunConfig:
    values = {**BASE, **override}
    if chunking is not None:
        values["chunking"] = chunking
    return RunConfig.model_validate(dict(pipeline={"id": "builtin_rag", "params": values}, top_k=5,
                                         judge_model="qwen2.5:7b-instruct", eval_split="val", is_dev=False))


def other(name):
    chunking, overlap = OTHERS[name]
    return run_config(chunking, overlap=overlap)


# ---------------------------------------------------------------- default = absent

def test_explicit_fixed_serializes_like_absent():
    assert params({"strategy": "fixed"}).model_dump(mode="json") == params().model_dump(mode="json")
    assert "chunking" not in params({"strategy": "fixed"}).model_dump(mode="json")


def test_absent_chunking_leaves_no_key_in_the_serialized_params():
    # a `chunking: None` key would change the config_hash of every existing config
    assert "chunking" not in params().model_dump(mode="json")
    assert "chunking" not in load_run_config(CONFIGS / "baseline.yaml").hash_payload()["pipeline"]["params"]


def test_explicit_fixed_keeps_the_config_hash():
    assert run_config({"strategy": "fixed"}).config_hash() == run_config().config_hash()


@pytest.mark.parametrize("name", list(OTHERS))
def test_other_strategies_have_their_own_hash(name):
    hashes = {run_config().config_hash()} | {other(n).config_hash() for n in OTHERS}
    assert len(hashes) == 1 + len(OTHERS)
    assert "chunking" in other(name).hash_payload()["pipeline"]["params"]


def test_window_and_stride_are_hashed():
    a = run_config(SLIDING, overlap=0).config_hash()
    assert run_config({**SLIDING, "window": 4}, overlap=0).config_hash() != a
    assert run_config({**SLIDING, "window": 3, "stride": 2}, overlap=0).config_hash() != a


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("chunking,overlap,message", [
    ({"strategy": "semantic"}, 50, "strategy"),
    ({"strategy": "recursive", "window": 3}, 50, "Extra inputs"),
    ({"strategy": "sliding_window", "window": 3}, 0, "stride"),
    ({"strategy": "sliding_window", "window": 2, "stride": 3}, 0, "must not be larger than window"),
    ({"strategy": "sliding_window", "window": 0, "stride": 1}, 0, "greater than or equal to 1"),
    ({"strategy": "sliding_window", "window": 21, "stride": 1}, 0, "less than or equal to 20"),
    (dict(SLIDING), 50, "does not use overlap"),
    ("recursive", 50, "chunking"),
])
def test_bad_chunking_is_rejected(chunking, overlap, message):
    with pytest.raises(ValidationError, match=message):
        params(chunking, overlap=overlap)


# ---------------------------------------------------------------- registry matches the chunker

MODELS = get_args(get_args(ChunkingParams)[0])


def strategy_name(model) -> str:
    return get_args(model.model_fields["strategy"].annotation)[0]


def test_every_strategy_has_a_params_model_and_back():
    assert {strategy_name(m) for m in MODELS} == set(STRATEGIES)


@pytest.mark.parametrize("model", MODELS, ids=[strategy_name(m) for m in MODELS])
def test_params_model_matches_the_strategy(model):
    spec = STRATEGIES[strategy_name(model)]
    assert tuple(f for f in model.model_fields if f != "strategy") == spec.options
    assert model.uses_overlap == spec.uses_overlap


# ---------------------------------------------------------------- params -> ChunkerConfig

def test_chunker_kwargs_off_is_the_pre_strategy_call():
    assert chunker_kwargs(params()) == {"chunk_size": 500, "overlap": 50}
    assert chunker_kwargs(params({"strategy": "fixed"})) == {"chunk_size": 500, "overlap": 50}


def test_chunker_kwargs_carry_strategy_and_options():
    assert chunker_kwargs(params({"strategy": "recursive"})) == {"chunk_size": 500, "overlap": 50, "strategy": "recursive"}
    assert chunker_kwargs(params(SLIDING, overlap=0)) == {
        "chunk_size": 500, "overlap": 0, "strategy": "sliding_window", "options": {"window": 3, "stride": 1}}


@pytest.mark.parametrize("name", list(OTHERS))
def test_chunker_kwargs_build_a_valid_chunker_config(name):
    chunking, overlap = OTHERS[name]
    cfg = ChunkerConfig(**chunker_kwargs(params(chunking, overlap=overlap)))
    assert cfg.strategy == name


class FakeChunkerConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def factory_stubs(monkeypatch):
    seen = {}

    class LooseSystem:
        def __init__(self, chunker_config, **kwargs):
            seen["chunker"] = chunker_config

    monkeypatch.setitem(sys.modules, "ragregress.rag.builtin_system", types.SimpleNamespace(BuiltinRAGSystem=LooseSystem))
    monkeypatch.setitem(sys.modules, "ragregress.rag.chunker", types.SimpleNamespace(ChunkerConfig=FakeChunkerConfig))
    return seen


def test_factory_off_makes_the_pre_strategy_chunker_call(factory_stubs):
    _build_builtin_rag(params())
    assert factory_stubs["chunker"].kwargs == {"chunk_size": 500, "overlap": 50}


def test_factory_passes_strategy_and_options(factory_stubs):
    _build_builtin_rag(params(SLIDING, overlap=0))
    assert factory_stubs["chunker"].kwargs["strategy"] == "sliding_window"
    assert factory_stubs["chunker"].kwargs["options"] == {"window": 3, "stride": 1}


# ---------------------------------------------------------------- collection names

def test_default_fingerprint_is_unchanged():
    assert _chunker_figerprint(ChunkerConfig(chunk_size=500, overlap=50)) == DEFAULT_FINGERPRINT


def test_explicit_fixed_has_the_default_fingerprint():
    assert _chunker_figerprint(ChunkerConfig(chunk_size=500, overlap=50, strategy="fixed")) == DEFAULT_FINGERPRINT


def test_each_strategy_gets_its_own_collection():
    configs = [ChunkerConfig(chunk_size=500, overlap=50)] + [
        ChunkerConfig(**chunker_kwargs(params(chunking, overlap=overlap))) for chunking, overlap in OTHERS.values()]
    names = [collection_name("all-MiniLM-L6-v2", c) for c in configs]
    assert len(set(names)) == len(configs)
    assert all(CHROMA_NAME.fullmatch(n) for n in names)


def test_option_order_does_not_change_the_collection():
    a = ChunkerConfig(chunk_size=500, overlap=0, strategy="sliding_window", options={"window": 3, "stride": 1})
    b = ChunkerConfig(chunk_size=500, overlap=0, strategy="sliding_window", options={"stride": 1, "window": 3})
    assert _chunker_figerprint(a) == _chunker_figerprint(b)


def test_window_and_stride_change_the_collection():
    def fp(window, stride):
        return _chunker_figerprint(ChunkerConfig(chunk_size=500, overlap=0, strategy="sliding_window",
                                                 options={"window": window, "stride": stride}))
    assert len({fp(3, 1), fp(3, 2), fp(4, 1)}) == 3


# ---------------------------------------------------------------- the shipped configs

def flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


@pytest.mark.parametrize("file,strategy,differs", [
    ("chunk_recursive.yaml", "recursive", {"pipeline.params.chunking.strategy"}),
    ("chunk_sentence.yaml", "sentence", {"pipeline.params.chunking.strategy"}),
    ("chunk_sliding_3_1.yaml", "sliding_window", {
        "pipeline.params.chunking.strategy", "pipeline.params.chunking.window",
        "pipeline.params.chunking.stride", "pipeline.params.overlap"}),
])
def test_chunking_configs_differ_from_baseline_only_in_chunking(file, strategy, differs):
    base, cfg = load_run_config(CONFIGS / "baseline.yaml"), load_run_config(CONFIGS / file)
    a, b = flatten(base.hash_payload()), flatten(cfg.hash_payload())
    assert cfg.pipeline.params.chunking.strategy == strategy
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == differs
    assert cfg.config_hash() != base.config_hash()
    assert (cfg.eval_split, cfg.is_dev) == ("val", False)