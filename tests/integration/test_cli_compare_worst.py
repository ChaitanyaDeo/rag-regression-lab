"""
tests/integration/test_cli_compare_worst.py

`ragregress compare --worst N` through Typer's CliRunner on a temporary SQLite database.
Checks: off by default, printed after the verdict when on, never changes the exit code, bad values rejected.

Run (repo root):  python -m pytest tests/integration/test_cli_compare_worst.py -v
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pytest
from typer.testing import CliRunner

from ragregress import cli
from ragregress.cli import app
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

runner = CliRunner()


# ---------------------------------------------------------------- builders

@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test_cli_compare_worst.db"
    init_db(path)
    return path


def _config() -> Dict[str, Any]:
    return {
        "eval_split": "val",
        "judge_model": "qwen2.5:7b",
        "top_k": 5,
        "pipeline": {"name": "baseline", "params": {"chunk_size": 500}},
    }


def add_run(db, run_id: str) -> None:
    Repository(db).create_run(
        Run(run_id=run_id, created_at="2026-10-19T00:00:00", is_dev=False, k=5, max_workers=1,
            config=_config(), config_hash="hash")
    )


def judged(run_id: str, qa_id: str, recall: float = 1.0, category: str = "cat") -> QueryResult:
    return QueryResult(
        run_id=run_id, qa_id=qa_id, question=f"q {qa_id}", query_status="judged", category=category,
        failure_mode="pass", generation_answer="an answer",
        metric_results={"recall_at_k": {"value": recall, "reason": ""}, "mrr": {"value": 1.0, "reason": ""}},
    )


def evaluator_failure(run_id: str, qa_id: str) -> QueryResult:
    return QueryResult(run_id=run_id, qa_id=qa_id, query_status="failed", failure_mode="judge_failed",
                       generation_answer="an answer", last_error="judge crashed")


def ids(n: int = 10) -> List[str]:
    return [f"q{i:02d}" for i in range(n)]


def seed(db, cand_rows: Iterable[QueryResult], n: int = 10) -> None:
    add_run(db, "base")
    add_run(db, "cand")
    Repository(db).save_query_results([judged("base", q) for q in ids(n)])
    Repository(db).save_query_results(list(cand_rows))


def seed_regression(db) -> None:
    """recall_at_k drops 1.0 -> 0.0 on q00..q04 (5 of 10 rows): verdict FAIL, equal worsening -> qa_id order."""
    seed(db, [judged("cand", q, recall=0.0 if k < 5 else 1.0) for k, q in enumerate(ids())])


def seed_small_drop(db) -> None:
    """One row drops by 0.5 -> mean drop 0.05 (exactly the threshold): verdict PASS, but one row did get worse."""
    seed(db, [judged("cand", q, recall=0.5 if q == "q03" else 1.0) for q in ids()])


def compare(db, *extra: str):
    return runner.invoke(app, ["compare", "--baseline", "base", "--candidate", "cand", "--db", str(db), *extra])


# ---------------------------------------------------------------- default: off

def test_diagnostics_are_off_by_default(db):
    seed_regression(db)
    result = compare(db)

    assert result.exit_code == 6, result.output
    assert "VERDICT: FAIL" in result.output
    assert "worst regressions" not in result.output
    assert "by category" not in result.output


def test_worst_zero_is_the_same_as_off(db):
    seed_regression(db)
    assert compare(db, "--worst", "0").output == compare(db).output


# ---------------------------------------------------------------- on

def test_worst_prints_the_block_after_the_verdict_and_keeps_exit_6(db):
    seed_regression(db)
    result = compare(db, "--worst", "3")
    out = result.output

    assert result.exit_code == 6, out
    assert "worst regressions: 5 of 10 common rows are worse on >= 1 gated metric (top 3 per metric)" in out
    assert out.index("VERDICT: FAIL") < out.index("worst regressions")          # printed AFTER the verdict
    assert "recall_at_k  (higher is better; 5 worse / 0 better / 5 tie of 10 compared)" in out
    assert "by category" in out and "by failure mode (candidate tag)" in out


def test_worst_respects_n(db):
    seed_regression(db)
    out = compare(db, "--worst", "2").output

    assert "q00" in out and "q01" in out
    assert "q02" not in out                                                      # cut at N, ties ordered by qa_id
    assert "(top 2 per metric)" in out


def test_worst_on_a_pass_verdict_still_exits_0(db):
    seed_small_drop(db)
    result = compare(db, "--worst", "5")

    assert result.exit_code == 0, result.output
    assert "VERDICT: PASS" in result.output
    assert "1 of 10 common rows are worse on >= 1 gated metric" in result.output
    assert "q03" in result.output


def test_worst_with_inconclusive_verdict_still_exits_7(db):
    seed(db, [judged("cand", q, recall=0.0) for q in ids()[:5]] + [evaluator_failure("cand", q) for q in ids()[5:]])
    result = compare(db, "--worst", "3")

    assert result.exit_code == 7, result.output
    assert "VERDICT: INCONCLUSIVE" in result.output
    assert "worst regressions" in result.output


# ---------------------------------------------------------------- bad input and failures

def test_negative_worst_is_a_typer_usage_error_exit_2(db):
    seed_regression(db)
    result = compare(db, "--worst", "-1")
    assert result.exit_code == 2


def test_non_integer_worst_is_a_usage_error_exit_2(db):
    seed_regression(db)
    assert compare(db, "--worst", "many").exit_code == 2


def test_error_inside_diagnostics_exits_1_like_any_unexpected_error(db, monkeypatch):
    seed_regression(db)

    def boom(*args, **kwargs):
        raise RuntimeError("diagnostics kaboom")

    monkeypatch.setattr(cli, "diagnose", boom)
    result = compare(db, "--worst", "3")

    assert result.exit_code == 1
    assert "unexpected RuntimeError: diagnostics kaboom" in result.output


def test_invalid_input_with_worst_still_exits_8(db):
    add_run(db, "base")
    result = runner.invoke(app, ["compare", "--baseline", "base", "--candidate", "ghost", "--db", str(db), "--worst", "3"])
    assert result.exit_code == 8
    assert "worst regressions" not in result.output


def test_help_lists_worst():
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--worst" in result.output