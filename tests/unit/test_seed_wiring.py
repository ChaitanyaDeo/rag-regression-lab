"""
tests/unit/test_seed_wiring.py

D10-B11: fixed seeds for the generator and the judge (D10-Q6a, option C).
  - pipeline.params.seed -> BuiltinRAGParams.seed -> GeneratorConfig.seed -> ChatOllama(seed=)
  - judge_seed           -> RunConfig.judge_seed  -> InstructorStructuredLLM(seed=) -> create(seed=)
Both default to 42 and are part of config_hash.

No Ollama, no models: LLM clients are replaced by fakes that record their kwargs.

D13-B2c: build_metrics now goes through ragregress.evaluation.judge.build_judge, which also passes
provider / base_url / api_key; the FakeJudge below accepts them (F-D12-23).
The judge's own wiring (providers, endpoints, keys) is tested in tests/unit/test_judge.py.

Run (repo root):  python -m pytest tests/unit/test_seed_wiring.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragregress.core.config import RunConfig, load_run_config
from ragregress.rag import generator as generator_mod
from ragregress.rag import no_rag_baseline as no_rag_mod
from ragregress.rag.generator import GeneratorConfig
from ragregress.rag.registry import BuiltinRAGParams, _build_builtin_rag

VAL_CONFIGS = [Path("configs/baseline.yaml"), Path("configs/chunk_1000.yaml")]


# ---------------------------------------------------------------- helpers

def params(**overrides):
    base = dict(embedding_model="all-MiniLM-L6-v2", chunk_size=500, overlap=50,
                generation_model="qwen2.5:3b-instruct")
    base.update(overrides)
    return base


def run_config(judge_seed=None, **param_overrides) -> RunConfig:
    raw = {
        "pipeline": {"id": "builtin_rag", "params": params(**param_overrides)},
        "top_k": 5,
        "judge_model": "qwen2.5:7b-instruct",
        "eval_split": "val",
        "is_dev": False,
    }
    if judge_seed is not None:
        raw["judge_seed"] = judge_seed
    return RunConfig.model_validate(raw)


class FakeChatOllama:
    """Records constructor kwargs; callable so `prompt | llm` coerces it to a Runnable."""
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeChatOllama.instances.append(self)

    def __call__(self, _prompt_value):
        return types.SimpleNamespace(content="an answer [source: p0000001]")


@pytest.fixture()
def fake_ollama(monkeypatch):
    FakeChatOllama.instances = []
    monkeypatch.setattr(generator_mod, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(no_rag_mod, "ChatOllama", FakeChatOllama)
    return FakeChatOllama


# ---------------------------------------------------------------- generator side

def test_generator_config_default_seed_is_42():
    assert GeneratorConfig().seed == 42


def test_run_generation_passes_seed_to_chat_ollama(fake_ollama):
    generator_mod.run_generation("q?", [], GeneratorConfig(seed=7))
    assert fake_ollama.instances[-1].kwargs == {
        "model": "qwen2.5:3b-instruct", "temperature": 0, "seed": 7,
    }


def test_no_rag_generation_passes_seed_to_chat_ollama(fake_ollama):
    no_rag_mod.run_generation_no_context("q?", GeneratorConfig(seed=9))
    assert fake_ollama.instances[-1].kwargs["seed"] == 9


def test_builtin_params_seed_defaults_to_42():
    assert BuiltinRAGParams(**params()).seed == 42


def test_builtin_params_seed_accepts_explicit_value():
    assert BuiltinRAGParams(**params(seed=7)).seed == 7


def test_builtin_params_seed_rejects_negative():
    with pytest.raises(ValidationError):
        BuiltinRAGParams(**params(seed=-1))


def test_factory_passes_params_seed_into_generator_config(monkeypatch):
    captured = {}

    class FakeSystem:
        def __init__(self, chunker_config, generator_config=None, embedding_model=None):
            captured["generator_config"] = generator_config

    class FakeChunkerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Stub the lazily imported heavy modules (sentence-transformers / chroma).
    monkeypatch.setitem(sys.modules, "ragregress.rag.builtin_system",
                        types.SimpleNamespace(BuiltinRAGSystem=FakeSystem))
    monkeypatch.setitem(sys.modules, "ragregress.rag.chunker",
                        types.SimpleNamespace(ChunkerConfig=FakeChunkerConfig))

    _build_builtin_rag(BuiltinRAGParams(**params(seed=11)))

    gc = captured["generator_config"]
    assert (gc.model_name, gc.seed) == ("qwen2.5:3b-instruct", 11)


# ---------------------------------------------------------------- judge side

class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "parsed"


@pytest.fixture()
def fake_instructor(monkeypatch):
    completions = FakeCompletions()
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    fake = types.SimpleNamespace(
        Mode=types.SimpleNamespace(JSON="json"),
        from_provider=lambda name, mode, **kwargs: client,      # base_url / api_key arrive only when set (D13-B2c)
    )
    monkeypatch.setitem(sys.modules, "instructor", fake)
    return completions


def test_judge_sends_seed_when_set(fake_instructor):
    from ragregress.evaluation.structured_llm import InstructorStructuredLLM

    InstructorStructuredLLM("qwen2.5:7b-instruct", seed=42).generate_structured("p", dict)
    call = fake_instructor.calls[-1]
    assert call["seed"] == 42
    assert call["temperature"] == 0


def test_judge_sends_no_seed_when_none(fake_instructor):
    from ragregress.evaluation.structured_llm import InstructorStructuredLLM

    InstructorStructuredLLM("qwen2.5:7b-instruct").generate_structured("p", dict)
    assert "seed" not in fake_instructor.calls[-1]


def test_build_metrics_passes_judge_seed(monkeypatch):
    import ragregress.evaluation.structured_llm as sl
    from ragregress.core.runner import build_metrics

    seen = {}

    class FakeJudge:
        def __init__(self, model_name, provider="ollama", max_retries=2, seed=None, base_url=None, api_key=None):
            seen.update(model_name=model_name, provider=provider, max_retries=max_retries, seed=seed)

    monkeypatch.setattr(sl, "InstructorStructuredLLM", FakeJudge)
    build_metrics(run_config(judge_seed=5))
    assert seen == {"model_name": "qwen2.5:7b-instruct", "provider": "ollama", "max_retries": 3, "seed": 5}


# ---------------------------------------------------------------- config / hash

def test_run_config_judge_seed_defaults_to_42():
    assert run_config().judge_seed == 42


def test_run_config_judge_seed_rejects_negative():
    with pytest.raises(ValidationError):
        run_config(judge_seed=-1)


def test_both_seeds_are_in_hash_payload():
    payload = run_config().hash_payload()
    assert payload["judge_seed"] == 42
    assert payload["pipeline"]["params"]["seed"] == 42


def test_changing_generator_seed_changes_hash():
    assert run_config().config_hash() != run_config(seed=43).config_hash()


def test_changing_judge_seed_changes_hash():
    assert run_config().config_hash() != run_config(judge_seed=43).config_hash()


def test_explicit_default_seeds_hash_same_as_omitted():
    assert run_config().config_hash() == run_config(judge_seed=42, seed=42).config_hash()


@pytest.mark.parametrize("path", VAL_CONFIGS, ids=lambda p: p.name)
def test_val_configs_write_both_seeds_explicitly(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["judge_seed"] == 42
    assert raw["pipeline"]["params"]["seed"] == 42
    cfg = load_run_config(path)
    assert (cfg.judge_seed, cfg.pipeline.params.seed) == (42, 42)