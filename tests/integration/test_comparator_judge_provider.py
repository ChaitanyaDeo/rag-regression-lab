"""
tests/integration/test_comparator_judge_provider.py

Day 13 (B2c): the judge is the pair (judge_provider, judge_model). The same model name on a different
service is a different judge, so faithfulness must not be gated across it.

A stored config WITHOUT judge_provider means "ollama": RunConfig leaves the default out of its dump,
and every run written before the field existed has no key.

Real (temporary) SQLite through the project's Repository. No LLM, no pipeline run. The real
ragregress.db is never touched.

Run (repo root):  python -m pytest tests/integration/test_comparator_judge_provider.py -v
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from ragregress.evaluation.comparator import compare_runs
from ragregress.storage.database import init_db
from ragregress.storage.models import Run
from ragregress.storage.repositories import Repository


@pytest.fixture()
def repo(tmp_path) -> Repository:
    db = tmp_path / "test_comparator_judge_provider.db"
    init_db(db)
    return Repository(db)


def config(judge_model: str = "qwen2.5:7b", judge_provider: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """Shape of a RunConfig dump. judge_provider is left out when None (= the default, as RunConfig does)."""
    cfg: Dict[str, Any] = {
        "eval_split": "val", "judge_model": judge_model, "top_k": 5,
        "pipeline": {"id": "builtin_rag", "params": {"chunk_size": 500}},
    }
    if judge_provider is not None:
        cfg["judge_provider"] = judge_provider
    cfg.update(extra)
    return cfg


def add_run(repo: Repository, run_id: str, cfg: Dict[str, Any]) -> None:
    repo.create_run(Run(
        run_id=run_id, created_at="2026-10-29T00:00:00", is_dev=False, k=5, max_workers=1,
        config=cfg, config_hash="hash",
    ))


def compare(repo: Repository, base_cfg: Dict[str, Any], cand_cfg: Dict[str, Any]):
    add_run(repo, "base", base_cfg)
    add_run(repo, "cand", cand_cfg)
    return compare_runs(repo, "base", "cand")


def test_same_model_name_on_a_different_provider_ungates_faithfulness(repo):
    result = compare(repo, config("gpt-4o-mini", "ollama"), config("gpt-4o-mini", "openai"))

    assert set(result.ungated) == {"faithfulness"}
    assert result.ungated["faithfulness"] == "judge_provider differs (baseline='ollama', candidate='openai')"
    assert result.warnings == [
        "judge_provider differs (baseline='ollama', candidate='openai'); faithfulness is not gated"
    ]
    assert "judge_model differs" not in result.ungated["faithfulness"]


def test_a_model_only_difference_keeps_the_original_message(repo):
    result = compare(repo, config("qwen2.5:7b"), config("llama3:8b"))

    assert result.ungated == {"faithfulness": "judge_model differs (baseline='qwen2.5:7b', candidate='llama3:8b')"}
    assert result.warnings == [
        "judge_model differs (baseline='qwen2.5:7b', candidate='llama3:8b'); faithfulness is not gated"
    ]


def test_model_and_provider_both_different_gives_two_reasons_in_one_warning(repo):
    result = compare(repo, config("qwen2.5:7b"), config("gpt-4o-mini", "openai"))

    reason = result.ungated["faithfulness"]
    assert reason == ("judge_model differs (baseline='qwen2.5:7b', candidate='gpt-4o-mini'); "
                      "judge_provider differs (baseline='ollama', candidate='openai')")
    assert result.warnings == [f"{reason}; faithfulness is not gated"]


def test_a_missing_provider_counts_as_ollama(repo):
    # omitted vs explicit default: the same judge
    same = compare(repo, config("qwen2.5:7b"), config("qwen2.5:7b", "ollama"))
    assert same.warnings == [] and same.ungated == {}


def test_a_missing_provider_against_openai_is_a_difference_reported_as_ollama(repo):
    result = compare(repo, config("gpt-4o-mini"), config("gpt-4o-mini", "openai"))
    assert result.ungated["faithfulness"] == "judge_provider differs (baseline='ollama', candidate='openai')"


def test_identical_judges_on_openai_have_no_warnings(repo):
    result = compare(repo, config("gpt-4o-mini", "openai"), config("gpt-4o-mini", "openai"))
    assert result.warnings == [] and result.ungated == {}


def test_the_provider_difference_shows_up_in_the_config_diff(repo):
    result = compare(repo, config("gpt-4o-mini"), config("gpt-4o-mini", "openai"))
    (change,) = [c for c in result.config_changes if c.key == "judge_provider"]
    assert str(change.before) == "—" and change.after == "openai"     # absent on the baseline side


def test_legacy_runs_without_a_judge_model_are_not_judged_on_the_provider(repo):
    add_run(repo, "base", {"is_dev": False, "k": 5})                       # legacy: no judge fields at all
    add_run(repo, "cand", config("gpt-4o-mini", "openai"))
    result = compare_runs(repo, "base", "cand")

    assert "faithfulness" not in result.ungated
    assert not any("differs" in w for w in result.warnings)     # only the legacy warning is present
    assert result.legacy_run_ids == ("base",)


def test_provider_difference_does_not_ungate_recall(repo):
    result = compare(repo, config("m", "ollama"), config("m", "openai"))
    assert "recall_at_k" not in result.ungated