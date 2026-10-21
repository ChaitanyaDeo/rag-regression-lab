"""
tests/integration/test_cli_compare.py

`ragregress compare` end to end through Typer's CliRunner, on a temporary SQLite
database: stored runs -> compare_runs -> assess_regression -> printed report -> process exit code.

Exit codes under test: 0 PASS | 6 FAIL | 7 INCONCLUSIVE | 8 invalid input
| 1 unexpected error | 2 reserved for Typer usage errors.

Run (repo root):  python -m pytest tests/integration/test_cli_compare.py -v
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
 
 
# ---------------------------------------------------------------- fixtures / builders
 
@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test_cli_compare.db"
    init_db(path)
    return path
 
 
def new_config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "eval_split": "val",
        "judge_model": "qwen2.5:7b",
        "top_k": 5,
        "pipeline": {"name": "baseline", "params": {"chunk_size": 500}},
    }
    cfg.update(overrides)
    return cfg
 
 
def add_run(db, run_id: str, *, config=None, k: int = 5, is_dev: bool = False, config_hash="hash") -> None:
    Repository(db).create_run(
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
 
 
def judged(run_id: str, qa_id: str, recall: Optional[float] = 1.0, mrr: Optional[float] = 1.0) -> QueryResult:
    return QueryResult(
        run_id=run_id, qa_id=qa_id, question=f"q {qa_id}", query_status="judged", category="cat",
        generation_answer="an answer",
        metric_results={"recall_at_k": {"value": recall, "reason": ""}, "mrr": {"value": mrr, "reason": ""}},
    )
 
 
def evaluator_failure(run_id: str, qa_id: str) -> QueryResult:
    return QueryResult(
        run_id=run_id, qa_id=qa_id, query_status="failed", failure_mode="judge_failed",
        generation_answer="an answer", last_error="judge crashed",
    )
 
 
def save(db, rows: Iterable[QueryResult]) -> None:
    Repository(db).save_query_results(list(rows))
 
 
def ids(n: int = 10) -> List[str]:
    return [f"q{i:02d}" for i in range(n)]
 
 
def seed(db, cand_rows: List[QueryResult], *, base_recall: float = 1.0, n: int = 10, **cand_run_kwargs) -> None:
    """Run `base` (n rows, every metric = base_recall) and run `cand` (given rows)."""
    add_run(db, "base")
    add_run(db, "cand", **cand_run_kwargs)
    save(db, [judged("base", q, recall=base_recall, mrr=base_recall) for q in ids(n)])
    save(db, cand_rows)
 
 
def compare(db, baseline: str = "base", candidate: str = "cand", *extra: str):
    return runner.invoke(
        app, ["compare", "--baseline", baseline, "--candidate", candidate, "--db", str(db), *extra]
    )
 
 
# ---------------------------------------------------------------- 0 / 6 / 7: the three verdicts
 
def test_pass_exits_0_and_prints_the_report(db):
    seed(db, [judged("cand", q) for q in ids()])
    result = compare(db)
 
    assert result.exit_code == 0, result.output
    out = result.output
    assert "VERDICT: PASS" in out
    assert "base" in out and "cand" in out                      # run header
    assert "recall_at_k" in out and "mrr" in out                # metric table
    assert "10/10 (100%)" in out                                # validity line
    assert "gated" in out
 
 
def test_regression_exits_6_and_names_the_metric(db):
    seed(db, [judged("cand", q, recall=0.0, mrr=1.0) for q in ids()])      # recall 1.0 -> 0.0
    result = compare(db)
 
    assert result.exit_code == 6, result.output
    assert "VERDICT: FAIL" in result.output
    assert "recall_at_k worsened by 1.0000" in result.output
    assert "regressed" in result.output                         # status column
 
 
def test_small_drop_within_threshold_still_exits_0(db):
    # 10 rows: one row drops 1.0 -> 0.5 => mean recall 0.95 vs 1.0 = exactly the 0.05 threshold (not regressed)
    rows = [judged("cand", q) for q in ids()]
    rows[0] = judged("cand", "q00", recall=0.5)
    seed(db, rows)
    result = compare(db)
    assert result.exit_code == 0, result.output
 
 
def test_low_validity_exits_7_even_when_a_metric_also_regressed(db):
    rows = [judged("cand", q, recall=0.0) for q in ids()[:5]] + [evaluator_failure("cand", q) for q in ids()[5:]]
    seed(db, rows)
    result = compare(db)
 
    assert result.exit_code == 7, result.output
    assert "VERDICT: INCONCLUSIVE" in result.output
    assert "candidate validity 5/10 = 50% < 80%" in result.output
    assert "also flagged regressed" in result.output            # INCONCLUSIVE beats FAIL, but the drop is shown
 
 
def test_no_common_rows_exits_7(db):
    add_run(db, "base")
    add_run(db, "cand")
    save(db, [judged("base", q) for q in ("a1", "a2", "a3")])
    save(db, [judged("cand", q) for q in ("b1", "b2", "b3")])
    result = compare(db)
    assert result.exit_code == 7, result.output
    assert "no rows measured in both runs" in result.output
 
 
# ---------------------------------------------------------------- report content
 
def test_config_changes_and_warnings_are_printed(db):
    seed(
        db,
        [judged("cand", q) for q in ids()],
        config=new_config(top_k=10, pipeline={"name": "baseline", "params": {"chunk_size": 1000}}),
        k=10,
    )
    result = compare(db)
 
    assert result.exit_code == 0, result.output
    assert "pipeline.params.chunk_size: 500 -> 1000" in result.output
    assert "top_k: 5 -> 10" in result.output
    assert "recall_at_k is not gated" in result.output
    assert "ungated" in result.output                           # status column for recall_at_k
 
 
def test_legacy_run_exits_0_with_warning_and_dash_marker(db):
    add_run(db, "base", config={"is_dev": False, "k": 5}, k=5, config_hash=None)     # legacy storage shape
    add_run(db, "cand")
    save(db, [judged("base", q) for q in ids()])
    save(db, [judged("cand", q) for q in ids()])
    result = compare(db)
 
    assert result.exit_code == 0, result.output
    assert "legacy run: config incomplete" in result.output
    assert "none (legacy)" in result.output.lower()             # run header shows the missing hash (case-insensitive)
    assert "—" in result.output
 
 
# ---------------------------------------------------------------- 8: invalid input
 
def test_unknown_candidate_exits_8(db):
    add_run(db, "base")
    result = compare(db, "base", "ghost")
    assert result.exit_code == 8
    assert "unknown run id" in result.output and "ghost" in result.output
 
 
def test_unknown_baseline_exits_8(db):
    add_run(db, "cand")
    result = compare(db, "ghost", "cand")
    assert result.exit_code == 8
    assert "ghost" in result.output
 
 
def test_dev_run_exits_8_with_an_explicit_message(db):
    add_run(db, "base")
    add_run(db, "dev1", is_dev=True)
    result = compare(db, "base", "dev1")
 
    assert result.exit_code == 8
    assert "dev run" in result.output
    assert "dev1" in result.output
    assert "VERDICT" not in result.output                       # an error, not a verdict
 
 
def test_dev_run_as_baseline_exits_8(db):
    add_run(db, "dev1", is_dev=True)
    add_run(db, "cand")
    assert compare(db, "dev1", "cand").exit_code == 8
 
 
def test_eval_split_mismatch_exits_8(db):
    add_run(db, "base", config=new_config(eval_split="val"))
    add_run(db, "cand", config=new_config(eval_split="test"))
    result = compare(db)
    assert result.exit_code == 8
    assert "eval_split" in result.output
 
 
def test_missing_database_exits_8_and_creates_nothing(tmp_path):
    missing = tmp_path / "nope.db"
    result = runner.invoke(app, ["compare", "--baseline", "a", "--candidate", "b", "--db", str(missing)])
 
    assert result.exit_code == 8
    assert "database not found" in result.output
    assert not missing.exists()                                 # sqlite must not create an empty file
 
 
# ---------------------------------------------------------------- 2 and 1
 
def test_missing_required_option_is_a_typer_usage_error_exit_2(db):
    result = runner.invoke(app, ["compare", "--baseline", "base", "--db", str(db)])
    assert result.exit_code == 2
 
 
def test_unexpected_exception_exits_1_without_a_traceback(db, monkeypatch):
    add_run(db, "base")
    add_run(db, "cand")
 
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")
 
    monkeypatch.setattr(cli, "compare_runs", boom)
    result = compare(db)
 
    assert result.exit_code == 1
    assert "unexpected RuntimeError: kaboom" in result.output
    assert "--debug" in result.output
    assert "Traceback" not in result.output
 
 
def test_debug_flag_prints_a_traceback_for_unexpected_errors(db, monkeypatch):
    add_run(db, "base")
    add_run(db, "cand")
 
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")
 
    monkeypatch.setattr(cli, "compare_runs", boom)
    result = compare(db, "base", "cand", "--debug")
 
    assert result.exit_code == 1
    assert "Traceback" in result.output
    assert "kaboom" in result.output
 
 
def test_help_lists_the_options():
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    for option in ("--baseline", "--candidate", "--db", "--debug"):
        assert option in result.output
 