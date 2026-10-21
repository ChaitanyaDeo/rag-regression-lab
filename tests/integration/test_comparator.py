"""
tests/integration/test_comparator.py

ragregress/evaluation/comparator.py against a real (temporary) SQLite database,
written and read through the project's own Repository. No LLM, no pipeline run.

Each test builds the tiny runs it needs inside tmp_path, so the real ragregress.db is never touched.

Run (repo root):  python -m pytest tests/integration/test_comparator.py -v
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import pytest

from ragregress.core.errors import (
    ComparisonInputError,
    DevRunRejectedError,
    IncompatibleRunsError,
    UnknownRunError,
)
from ragregress.evaluation.comparator import (
    MISSING,
    RowKind,
    compare_runs,
    load_snapshot,
)
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository


# ---------------------------------------------------------------- fixtures / builders

@pytest.fixture()
def repo(tmp_path) -> Repository:
    db = tmp_path / "test_comparator.db"
    init_db(db)
    return Repository(db)


def new_config(**overrides: Any) -> Dict[str, Any]:
    """Shape of a RunConfig dump (has a `pipeline` key -> not a legacy run)."""
    cfg: Dict[str, Any] = {
        "eval_split": "val",
        "judge_model": "qwen2.5:7b",
        "top_k": 5,
        "pipeline": {"name": "baseline", "params": {"chunk_size": 500}},
    }
    cfg.update(overrides)
    return cfg


def add_run(
    repo: Repository,
    run_id: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    k: int = 5,
    is_dev: bool = False,
    config_hash: Optional[str] = "hash",
) -> None:
    repo.create_run(
        Run(
            run_id=run_id,
            created_at="2026-10-17T00:00:00",
            is_dev=is_dev,
            k=k,
            max_workers=1,
            config=new_config(top_k=k) if config is None else config,
            config_hash=config_hash,
        )
    )


def add_legacy_run(repo: Repository, run_id: str, *, k: int = 5) -> None:
    # Legacy storage: config_json == {is_dev, k}, no config_hash
    add_run(repo, run_id, config={"is_dev": False, "k": k}, k=k, config_hash=None)


def judged(run_id: str, qa_id: str, category: str = "cat", **metrics: Optional[float]) -> QueryResult:
    return QueryResult(
        run_id=run_id,
        qa_id=qa_id,
        question=f"question {qa_id}",
        query_status="judged",
        category=category,
        generation_answer="an answer",
        metric_results={name: {"value": v, "reason": ""} for name, v in metrics.items()},
    )


def system_failure(run_id: str, qa_id: str) -> QueryResult:
    # failed, and the pipeline never produced an answer
    return QueryResult(
        run_id=run_id, qa_id=qa_id, query_status="failed", failure_mode="generate_failed",
        generation_answer=None, last_error="boom",
    )


def evaluator_failure(run_id: str, qa_id: str) -> QueryResult:
    # failed, but an answer exists: the judge crashed
    return QueryResult(
        run_id=run_id, qa_id=qa_id, query_status="failed", failure_mode="judge_failed",
        generation_answer="an answer", last_error="judge crashed",
    )


def save(repo: Repository, rows: Iterable[QueryResult]) -> None:
    repo.save_query_results(list(rows))


def seed_pair(repo: Repository, base_rows=None, cand_rows=None, **cand_kwargs) -> None:
    """Runs `base` and `cand` with default new-style configs, plus optional rows."""
    add_run(repo, "base")
    add_run(repo, "cand", **cand_kwargs)
    if base_rows:
        save(repo, base_rows)
    if cand_rows:
        save(repo, cand_rows)


def metric(result, name: str):
    return next(m for m in result.metrics if m.name == name)


# ---------------------------------------------------------------- row classification (D8-Q3)

def test_row_classification_and_values(repo):
    add_run(repo, "r")
    save(repo, [
        judged("r", "q1", category="easy", recall_at_k=1.0, mrr=0.5),
        judged("r", "q2", category="hard", recall_at_k=0.0, mrr=None),    # metric stored as null
        system_failure("r", "q3"),
        evaluator_failure("r", "q4"),
    ])

    snap = load_snapshot(repo, "r")

    assert snap.attempted == 4
    assert snap.applicable == frozenset({"recall_at_k", "mrr"})

    assert snap.rows["q1"].kind is RowKind.MEASURED
    assert snap.rows["q1"].values == {"recall_at_k": 1.0, "mrr": 0.5}
    assert snap.rows["q1"].category == "easy"

    assert snap.rows["q2"].kind is RowKind.MEASURED
    assert snap.rows["q2"].values == {"recall_at_k": 0.0, "mrr": None}

    # no answer -> system failure -> 0 on every applicable metric
    assert snap.rows["q3"].kind is RowKind.SYSTEM_FAILURE
    assert snap.rows["q3"].values == {"recall_at_k": 0.0, "mrr": 0.0}
    assert snap.rows["q3"].failure_mode == "generate_failed"

    # answer present -> evaluator failure -> unmeasured
    assert snap.rows["q4"].kind is RowKind.EVALUATOR_FAILURE
    assert snap.rows["q4"].values == {"recall_at_k": None, "mrr": None}


def test_nan_metric_value_is_unmeasured(repo):
    add_run(repo, "r")
    save(repo, [judged("r", "q1", faithfulness=float("nan")), judged("r", "q2", faithfulness=0.5)])
    snap = load_snapshot(repo, "r")
    assert snap.rows["q1"].values == {"faithfulness": None}
    assert snap.rows["q2"].values == {"faithfulness": 0.5}


def test_unrecognised_status_is_treated_as_unmeasured_not_zero(repo):
    add_run(repo, "r")
    save(repo, [
        judged("r", "q1", recall_at_k=1.0),
        QueryResult(run_id="r", qa_id="q2", query_status="pending", generation_answer=None),
    ])
    row = load_snapshot(repo, "r").rows["q2"]
    assert row.kind is RowKind.EVALUATOR_FAILURE
    assert row.values == {"recall_at_k": None}


def test_applicable_comes_only_from_judged_rows(repo):
    add_run(repo, "r")
    failed = system_failure("r", "q2")
    failed.metric_results = {"ghost": {"value": 1.0, "reason": ""}}     # should never happen; must be ignored
    save(repo, [judged("r", "q1", mrr=1.0), failed])
    assert load_snapshot(repo, "r").applicable == frozenset({"mrr"})


def test_run_with_no_judged_rows_has_no_applicable_metrics(repo):
    add_run(repo, "r")
    save(repo, [system_failure("r", "q1"), system_failure("r", "q2")])
    snap = load_snapshot(repo, "r")
    assert snap.applicable == frozenset()
    assert snap.attempted == 2
    assert all(r.values == {} for r in snap.rows.values())


# ---------------------------------------------------------------- hard rejections (exit 8)

def test_unknown_baseline_run_id(repo):
    add_run(repo, "cand")
    with pytest.raises(UnknownRunError) as exc:
        compare_runs(repo, "nope", "cand")
    assert exc.value.run_id == "nope"
    assert exc.value.exit_code == 8


def test_unknown_candidate_run_id(repo):
    add_run(repo, "base")
    with pytest.raises(UnknownRunError) as exc:
        compare_runs(repo, "base", "nope")
    assert exc.value.run_id == "nope"


def test_dev_run_rejected_as_baseline(repo):
    add_run(repo, "dev1", is_dev=True)
    add_run(repo, "cand")
    with pytest.raises(DevRunRejectedError) as exc:
        compare_runs(repo, "dev1", "cand")
    assert exc.value.role == "baseline"
    assert exc.value.run_id == "dev1"
    assert exc.value.exit_code == 8
    assert "dev run" in str(exc.value)


def test_dev_run_rejected_as_candidate(repo):
    add_run(repo, "base")
    add_run(repo, "dev1", is_dev=True)
    with pytest.raises(DevRunRejectedError) as exc:
        compare_runs(repo, "base", "dev1")
    assert exc.value.role == "candidate"


def test_eval_split_mismatch_is_rejected(repo):
    add_run(repo, "base", config=new_config(eval_split="val"))
    add_run(repo, "cand", config=new_config(eval_split="test"))
    with pytest.raises(IncompatibleRunsError) as exc:
        compare_runs(repo, "base", "cand")
    assert "eval_split" in str(exc.value)
    assert "val" in str(exc.value) and "test" in str(exc.value)
    assert exc.value.exit_code == 8


def test_every_rejection_is_a_comparison_input_error(repo):
    add_run(repo, "dev1", is_dev=True)
    add_run(repo, "cand")
    with pytest.raises(ComparisonInputError):
        compare_runs(repo, "dev1", "cand")
    with pytest.raises(ComparisonInputError):
        compare_runs(repo, "cand", "missing")


# ---------------------------------------------------------------- warnings + ungated (D8-Q6)

def test_identical_setup_has_no_warnings_and_nothing_ungated(repo):
    seed_pair(repo, [judged("base", "q1", recall_at_k=1.0)], [judged("cand", "q1", recall_at_k=1.0)])
    result = compare_runs(repo, "base", "cand")
    assert result.warnings == []
    assert result.ungated == {}
    assert result.config_changes == []
    assert result.legacy_run_ids == ()


def test_top_k_mismatch_warns_and_ungates_recall_only(repo):
    add_run(repo, "base", k=5)
    add_run(repo, "cand", k=10)
    result = compare_runs(repo, "base", "cand")

    assert set(result.ungated) == {"recall_at_k"}
    assert "top_k differs (baseline=5, candidate=10)" in result.ungated["recall_at_k"]
    assert any("top_k differs" in w and "recall_at_k is not gated" in w for w in result.warnings)
    assert "top_k" in {c.key for c in result.config_changes}


def test_judge_model_mismatch_warns_and_ungates_faithfulness_only(repo):
    add_run(repo, "base", config=new_config(judge_model="qwen2.5:7b"))
    add_run(repo, "cand", config=new_config(judge_model="llama3:8b"))
    result = compare_runs(repo, "base", "cand")

    assert set(result.ungated) == {"faithfulness"}
    assert "qwen2.5:7b" in result.ungated["faithfulness"] and "llama3:8b" in result.ungated["faithfulness"]
    assert any("faithfulness is not gated" in w for w in result.warnings)


def test_both_mismatches_ungate_both_metrics(repo):
    add_run(repo, "base", config=new_config(judge_model="a", top_k=5), k=5)
    add_run(repo, "cand", config=new_config(judge_model="b", top_k=10), k=10)
    result = compare_runs(repo, "base", "cand")
    assert set(result.ungated) == {"recall_at_k", "faithfulness"}
    assert len(result.warnings) == 2


def test_question_set_mismatch_warns_and_uses_common_rows(repo):
    seed_pair(
        repo,
        [judged("base", q, recall_at_k=1.0) for q in ("q1", "q2", "q3")],
        [judged("cand", q, recall_at_k=1.0) for q in ("q2", "q3", "q4", "q5")],
    )
    result = compare_runs(repo, "base", "cand")

    assert result.common_row_ids == ("q2", "q3")
    assert result.only_in_baseline == ("q1",)
    assert result.only_in_candidate == ("q4", "q5")
    assert any("question sets differ (1 only in baseline, 2 only in candidate)" in w for w in result.warnings)
    assert "2 common rows" in " ".join(result.warnings)
    assert metric(result, "recall_at_k").n_common == 2


# ---------------------------------------------------------------- config diff (D8-Q7)

def test_config_diff_lists_changed_dotted_keys_only(repo):
    add_run(repo, "base", config=new_config())
    add_run(repo, "cand", config=new_config(pipeline={"name": "baseline", "params": {"chunk_size": 1000}}))
    result = compare_runs(repo, "base", "cand")

    assert [(c.key, c.before, c.after) for c in result.config_changes] == [
        ("pipeline.params.chunk_size", 500, 1000)
    ]


def test_operational_keys_are_not_experiment_changes(repo):
    add_run(repo, "base", config=new_config(db_path="a.db", max_workers=1))
    add_run(repo, "cand", config=new_config(db_path="b.db", max_workers=8))
    assert compare_runs(repo, "base", "cand").config_changes == []


def test_key_present_on_one_side_only_is_marked_missing(repo):
    add_run(repo, "base", config=new_config())
    add_run(repo, "cand", config=new_config(rerank=True))
    (change,) = compare_runs(repo, "base", "cand").config_changes
    assert change.key == "rerank"
    assert change.before is MISSING
    assert change.after is True
    assert str(MISSING) == "—"


# ---------------------------------------------------------------- legacy runs (D8-Q2, Q7)

def test_legacy_baseline_is_allowed_with_a_warning(repo):
    add_legacy_run(repo, "base", k=5)
    add_run(repo, "cand", k=5)
    save(repo, [judged("base", "q1", recall_at_k=1.0), judged("cand", "q1", recall_at_k=0.0)])

    result = compare_runs(repo, "base", "cand")

    assert result.legacy_run_ids == ("base",)
    legacy_warnings = [w for w in result.warnings if "legacy run: config incomplete" in w]
    assert len(legacy_warnings) == 1 and "base" in legacy_warnings[0]
    assert metric(result, "recall_at_k").delta == pytest.approx(-1.0)     # still compared


def test_legacy_k_is_not_a_false_config_difference(repo):
    add_legacy_run(repo, "base", k=5)
    add_run(repo, "cand", k=5)
    keys = {c.key for c in compare_runs(repo, "base", "cand").config_changes}
    assert "k" not in keys
    assert "top_k" not in keys          # legacy `k` was renamed top_k and matches the candidate's top_k


def test_legacy_run_shows_missing_keys_as_dash(repo):
    add_legacy_run(repo, "base", k=5)
    add_run(repo, "cand", k=5)
    changes = {c.key: c for c in compare_runs(repo, "base", "cand").config_changes}

    assert changes["eval_split"].before is MISSING and changes["eval_split"].after == "val"
    assert changes["pipeline.params.chunk_size"].before is MISSING
    assert changes["pipeline.name"].before is MISSING


def test_legacy_top_k_difference_is_still_detected(repo):
    add_legacy_run(repo, "base", k=3)
    add_run(repo, "cand", k=5)
    result = compare_runs(repo, "base", "cand")
    assert "recall_at_k" in result.ungated
    top_k = next(c for c in result.config_changes if c.key == "top_k")
    assert (top_k.before, top_k.after) == (3, 5)


def test_legacy_run_does_not_trigger_the_eval_split_rejection(repo):
    # a legacy config has no eval_split, so there is nothing to mismatch
    add_legacy_run(repo, "base")
    add_run(repo, "cand", config=new_config(eval_split="test"))
    compare_runs(repo, "base", "cand")      # must not raise


def test_two_legacy_runs_are_both_named_in_one_warning(repo):
    add_legacy_run(repo, "old1")
    add_legacy_run(repo, "old2")
    result = compare_runs(repo, "old1", "old2")
    assert result.legacy_run_ids == ("old1", "old2")
    (warning,) = [w for w in result.warnings if "legacy run" in w]
    assert "old1" in warning and "old2" in warning


# ---------------------------------------------------------------- deltas (D8-Q3 b1)

def test_deltas_use_common_rows_only(repo):
    # common rows: q1..q3. Baseline recall 1,1,0 (q4=0 is baseline-only); candidate 1,0,0 (q5=1 is candidate-only)
    seed_pair(
        repo,
        [judged("base", "q1", recall_at_k=1.0), judged("base", "q2", recall_at_k=1.0),
         judged("base", "q3", recall_at_k=0.0), judged("base", "q4", recall_at_k=0.0)],
        [judged("cand", "q1", recall_at_k=1.0), judged("cand", "q2", recall_at_k=0.0),
         judged("cand", "q3", recall_at_k=0.0), judged("cand", "q5", recall_at_k=1.0)],
    )
    d = metric(compare_runs(repo, "base", "cand"), "recall_at_k")

    assert d.n_common == 3
    assert d.before == pytest.approx(2 / 3)
    assert d.after == pytest.approx(1 / 3)
    assert d.delta == pytest.approx(-1 / 3)
    assert (d.better, d.worse, d.tie) == (0, 1, 2)


def test_better_worse_tie_counts(repo):
    seed_pair(
        repo,
        [judged("base", "q1", mrr=0.2), judged("base", "q2", mrr=0.8), judged("base", "q3", mrr=0.5)],
        [judged("cand", "q1", mrr=0.9), judged("cand", "q2", mrr=0.1), judged("cand", "q3", mrr=0.5)],
    )
    d = metric(compare_runs(repo, "base", "cand"), "mrr")
    assert (d.better, d.worse, d.tie) == (1, 1, 1)


def test_system_failure_counts_as_zero_and_evaluator_failure_is_excluded(repo):
    seed_pair(
        repo,
        [judged("base", q, recall_at_k=1.0) for q in ("q1", "q2", "q3", "q4")],
        [judged("cand", "q1", recall_at_k=1.0),
         system_failure("cand", "q2"),            # -> 0.0, kept
         evaluator_failure("cand", "q3"),         # -> None, dropped
         judged("cand", "q4", recall_at_k=1.0)],
    )
    result = compare_runs(repo, "base", "cand")
    d = metric(result, "recall_at_k")

    assert d.n_common == 3                         # q1, q2, q4
    assert d.after == pytest.approx(2 / 3)
    assert d.before == pytest.approx(1.0)          # baseline mean is over the SAME three rows
    assert (d.better, d.worse, d.tie) == (0, 1, 2)
    assert result.candidate.attempted == 4         # failed rows stay in the denominator


def test_metric_absent_from_one_run_is_flagged_not_applicable(repo):
    seed_pair(
        repo,
        [judged("base", "q1", recall_at_k=1.0, faithfulness=0.8)],
        [judged("cand", "q1", recall_at_k=1.0)],
    )
    result = compare_runs(repo, "base", "cand")
    f = metric(result, "faithfulness")

    assert f.applicable_base is True and f.applicable_cand is False
    assert f.n_common == 0
    assert f.before is None and f.after is None and f.delta is None
    assert metric(result, "recall_at_k").applicable_cand is True


def test_metric_display_order(repo):
    seed_pair(
        repo,
        [judged("base", "q1", zeta=1.0, faithfulness=1.0, alpha=1.0, mrr=1.0, recall_at_k=1.0, token_prf1=1.0)],
        [judged("cand", "q1", zeta=1.0, faithfulness=1.0, alpha=1.0, mrr=1.0, recall_at_k=1.0, token_prf1=1.0)],
    )
    names = [m.name for m in compare_runs(repo, "base", "cand").metrics]
    assert names == ["recall_at_k", "mrr", "token_prf1", "faithfulness", "alpha", "zeta"]


def test_delta_is_nan_free_when_there_is_no_common_data(repo):
    seed_pair(repo, [judged("base", "q1", mrr=1.0)], [judged("cand", "q2", mrr=1.0)])
    d = metric(compare_runs(repo, "base", "cand"), "mrr")
    assert d.n_common == 0
    assert d.delta is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in (d.before, d.after, d.delta))