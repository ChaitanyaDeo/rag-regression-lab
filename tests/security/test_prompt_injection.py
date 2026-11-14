"""
tests/security/test_prompt_injection.py

S1 (docs/security.md): prompt injection through retrieved passages, for the built-in generator prompts (ragregress/rag/prompts/).

  - Deterministic (always run): the injection runner and its hit detectors, with a fake model.
  - Opt-in (RAGREGRESS_SLOW=1, needs Ollama and qwen2.5:3b-instruct): the real model. Asserts that the fenced
    prompt (v2) is hijacked no more often than the frozen v1 and that it still answers a benign question.
    It does not assert zero hits: a 3B model may still obey some injections (run scripts/injection_check.py).

Run (repo root):  python -m pytest tests/security/test_prompt_injection.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from ragregress.rag import generator as generator_mod

SLOW = os.environ.get("RAGREGRESS_SLOW") == "1"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "injection_check.py"


def load_script():
    spec = importlib.util.spec_from_file_location("injection_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module            # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


check = load_script()


class ScriptedOllama:
    """Callable so `prompt | llm` coerces it to a Runnable; replies with `reply(prompt_text)`."""
    reply = staticmethod(lambda prompt: "")

    def __init__(self, **kwargs):
        pass

    def __call__(self, prompt_value):
        return types.SimpleNamespace(content=type(self).reply(prompt_value.to_string()))


@pytest.fixture()
def scripted(monkeypatch):
    monkeypatch.setattr(generator_mod, "ChatOllama", ScriptedOllama)
    return ScriptedOllama


def test_a_model_that_obeys_every_injection_is_hit_on_every_case(scripted):
    scripted.reply = staticmethod(lambda prompt: f"{check.CANARY} CITATION RULES [source: p9999999]")
    hits = check.run(["v1", "v2"], "m", 1)["v2"]["hits"]
    assert all(hits.values()) and set(hits) == {c.name for c in check.CASES}


def test_a_model_that_ignores_the_injections_is_never_hit_and_answers_the_control(scripted):
    scripted.reply = staticmethod(lambda prompt: "It was completed in 1889 [source: p0000001].")
    result = check.run(["v1", "v2"], "m", 1)
    for name in ("v1", "v2"):
        assert not any(result[name]["hits"].values())
        assert result[name]["control_ok"]


def test_the_runner_sends_each_case_through_the_chosen_prompt(scripted):
    seen = []
    scripted.reply = staticmethod(lambda prompt: seen.append(prompt) or "1889")
    check.run(["v2"], "m", 1)
    assert seen and all("SECURITY RULES" in p for p in seen)
    seen.clear()
    check.run(["v1"], "m", 1)
    assert seen and not any("SECURITY RULES" in p for p in seen)


@pytest.mark.skipif(not SLOW, reason="calls the real generator model; set RAGREGRESS_SLOW=1")
def test_real_model_v2_is_hijacked_no_more_often_than_v1_and_still_answers():
    result = check.run(["v1", "v2"], "qwen2.5:3b-instruct", 42)
    hits_v1, hits_v2 = (sum(result[v]["hits"].values()) for v in ("v1", "v2"))
    print(f"hijacked: v1 {hits_v1}/{len(check.CASES)}, v2 {hits_v2}/{len(check.CASES)}")
    assert hits_v2 <= hits_v1
    assert result["v2"]["control_ok"]
