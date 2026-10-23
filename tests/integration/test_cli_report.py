"""
tests/integration/test_cli_report.py

`ragregress report` end to end through Typer's CliRunner, on a temporary SQLite database plus
temporary passages / split / targets files: stored runs -> generate_report -> HTML file -> exit code.

Exit codes under test: 0 report written (any verdict) | 8 invalid input | 1 unexpected error
| 2 Typer usage error.

Run (repo root):  python -m pytest tests/integration/test_cli_report.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest
from typer.testing import CliRunner

from ragregress import cli
from ragregress.cli import app
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

runner = CliRunner()

TARGETS_YAML = """\
source: README.md DEC-03 (2026-09-10)
targets:
  recall_at_k: {label: "Recall@5", low: 0.80, high: 0.85}
  mrr:         {label: "MRR", low: 0.65, high: 0.75}
"""


# ---------------------------------------------------------------- fixtures / builders

@pytest.fixture()
def files(tmp_path):
    db = tmp_path / "test_cli_report.db"
    init_db(db)
    passages = tmp_path / "passages.json"
    passages.write_text(json.dumps([{"passage_id": "p1", "title": "T1", "text": "Passage one."}]), encoding="utf-8")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "val.json").write_text(
        json.dumps([{"qa_id": f"q{i:02d}", "answer": [f"gold {i}"]} for i in range(10)]), encoding="utf-8")
    targets = tmp_path / "targets.yaml"
    targets.write_text(TARGETS_YAML, encoding="utf-8")
    return {"db": db, "passages": passages, "splits": splits, "targets": targets, "tmp": tmp_path,
            "out": tmp_path / "out" / "report.html"}


def add_run(db, run_id: str, *, is_dev: bool = False, split: str = "val") -> None:
    Repository(db).create_run(Run(
        run_id=run_id, created_at="2026-10-23T00:00:00", is_dev=is_dev, k=5, max_workers=1,
        config={"eval_split": split, "judge_model": "j", "pipeline": {"name": "b", "params": {"chunk_size": 500}}},
        config_hash=f"hash-{run_id}",
    ))


def judged(run_id: str, qa_id: str, recall: float = 1.0, mrr: float = 1.0) -> QueryResult:
    return QueryResult(
        run_id=run_id, qa_id=qa_id, question=f"q {qa_id}", query_status="judged", category="cat",
        generation_answer="an answer", retrieved_ids=["p1"], gold_ids=["p1"],
        metric_results={"recall_at_k": {"value": recall, "reason": ""}, "mrr": {"value": mrr, "reason": ""}},
    )


def evaluator_failure(run_id: str, qa_id: str) -> QueryResult:
    return QueryResult(run_id=run_id, qa_id=qa_id, query_status="failed", failure_mode="judge_failed",
                       generation_answer="an answer", last_error="judge crashed")


def ids(n: int = 10) -> List[str]:
    return [f"q{i:02d}" for i in range(n)]


def seed(db, cand_rows: List[QueryResult]) -> None:
    add_run(db, "base")
    add_run(db, "cand")
    repo = Repository(db)
    repo.save_query_results([judged("base", q) for q in ids()])
    repo.save_query_results(cand_rows)


def report(files, *extra: str, out: Optional[Path] = None, baseline="base", candidate="cand"):
    args = ["report", "--baseline", baseline, "--candidate", candidate, "--db", str(files["db"]),
            "--passages", str(files["passages"]), "--splits-dir", str(files["splits"])]
    if out is not False:
        args += ["--out", str(out or files["out"])]
    return runner.invoke(app, args + list(extra))


# ---------------------------------------------------------------- exit 0 for every verdict

def test_pass_writes_report_and_exits_0(files):
    seed(files["db"], [judged("cand", q) for q in ids()])
    result = report(files)

    assert result.exit_code == 0, result.output
    assert "verdict: PASS" in result.output
    assert f"report written: {files['out'].resolve()}" in result.output
    assert "warnings:" not in result.output
    html = files["out"].read_text(encoding="utf-8")
    assert 'class="banner verdict-pass"' in html
    assert "<td>Recall@5</td>" not in html                      # no --targets -> no target table


def test_fail_verdict_still_exits_0(files):
    seed(files["db"], [judged("cand", q, recall=0.0) for q in ids()])
    result = report(files)

    assert result.exit_code == 0, result.output                 # gating is compare's job (D9-Q15)
    assert "verdict: FAIL" in result.output
    assert 'class="banner verdict-fail"' in files["out"].read_text(encoding="utf-8")


def test_inconclusive_verdict_exits_0_with_warning_banner(files):
    seed(files["db"], [judged("cand", q) for q in ids()[:5]] + [evaluator_failure("cand", q) for q in ids()[5:]])
    result = report(files)

    assert result.exit_code == 0, result.output
    assert "verdict: INCONCLUSIVE" in result.output
    html = files["out"].read_text(encoding="utf-8")
    assert 'class="banner verdict-inconclusive"' in html
    assert 'class="banner verdict-pass"' not in html


# ---------------------------------------------------------------- options

def test_default_output_path_is_under_reports(files, monkeypatch):
    seed(files["db"], [judged("cand", q) for q in ids()])
    monkeypatch.chdir(files["tmp"])
    result = report(files, out=False)

    expected = files["tmp"] / "reports" / "base__vs__cand.html"
    assert result.exit_code == 0, result.output
    assert expected.is_file()
    assert f"report written: {expected.resolve()}" in result.output


def test_out_creates_parent_directories(files):
    seed(files["db"], [judged("cand", q) for q in ids()])
    nested = files["tmp"] / "a" / "b" / "c.html"
    result = report(files, out=nested)
    assert result.exit_code == 0, result.output
    assert nested.is_file()


def test_targets_option_adds_target_table(files):
    seed(files["db"], [judged("cand", q) for q in ids()])
    result = report(files, "--targets", str(files["targets"]))
    assert result.exit_code == 0, result.output
    html = files["out"].read_text(encoding="utf-8")
    assert "<td>Recall@5</td>" in html and "README.md DEC-03" in html


def test_warnings_count_is_printed(files):
    seed(files["db"], [judged("cand", q, mrr=0.0) for q in ids()])
    result = report(files, "--passages", str(files["tmp"] / "missing.json"))   # later option wins
    assert result.exit_code == 0, result.output
    assert "warnings: 1 (see report)" in result.output


def test_top_zero_is_a_usage_error(files):
    seed(files["db"], [judged("cand", q) for q in ids()])
    assert report(files, "--top", "0").exit_code == 2


def test_open_calls_browser_with_file_uri(files, monkeypatch):
    seed(files["db"], [judged("cand", q) for q in ids()])
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda uri: opened.append(uri) or True)
    result = report(files, "--open")

    assert result.exit_code == 0, result.output
    assert opened == [files["out"].resolve().as_uri()]
    assert "could not open a browser" not in result.output


def test_open_failure_is_a_note_not_an_error(files, monkeypatch):
    seed(files["db"], [judged("cand", q) for q in ids()])
    monkeypatch.setattr(cli.webbrowser, "open", lambda uri: False)
    result = report(files, "--open")

    assert result.exit_code == 0, result.output
    assert "note: could not open a browser" in result.output
    assert files["out"].is_file()


# ---------------------------------------------------------------- exit 8: invalid input, nothing written

def test_unknown_run_exits_8_and_writes_nothing(files):
    add_run(files["db"], "base")
    result = report(files, candidate="ghost")
    assert result.exit_code == 8
    assert "unknown run id" in result.output and "ghost" in result.output
    assert not files["out"].exists()


def test_dev_run_exits_8(files):
    add_run(files["db"], "base")
    add_run(files["db"], "dev1", is_dev=True)
    result = report(files, candidate="dev1")
    assert result.exit_code == 8
    assert "dev run" in result.output
    assert not files["out"].exists()


def test_eval_split_mismatch_exits_8(files):
    add_run(files["db"], "base", split="val")
    add_run(files["db"], "cand", split="test")
    result = report(files)
    assert result.exit_code == 8
    assert "eval_split" in result.output


def test_missing_database_exits_8_and_creates_nothing(files):
    missing = files["tmp"] / "nope.db"
    result = runner.invoke(app, ["report", "--baseline", "a", "--candidate", "b", "--db", str(missing),
                                 "--out", str(files["out"])])
    assert result.exit_code == 8
    assert "database not found" in result.output
    assert not missing.exists() and not files["out"].exists()


def test_bad_targets_file_exits_8_and_writes_nothing(files):
    seed(files["db"], [judged("cand", q) for q in ids()])
    bad = files["tmp"] / "bad.yaml"
    bad.write_text("targets:\n  mrr: {low: 0.9, high: 0.1}\n", encoding="utf-8")
    result = report(files, "--targets", str(bad))
    assert result.exit_code == 8
    assert "targets file" in result.output and "low 0.9 > high 0.1" in result.output
    assert not files["out"].exists()


def test_missing_targets_file_exits_8(files):
    seed(files["db"], [judged("cand", q) for q in ids()])
    result = report(files, "--targets", str(files["tmp"] / "nope.yaml"))
    assert result.exit_code == 8
    assert "not found" in result.output


# ---------------------------------------------------------------- 2 and 1

def test_missing_required_option_is_exit_2(files):
    result = runner.invoke(app, ["report", "--baseline", "base", "--db", str(files["db"])])
    assert result.exit_code == 2


def test_unexpected_exception_exits_1_without_traceback(files, monkeypatch):
    seed(files["db"], [judged("cand", q) for q in ids()])

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "generate_report", boom)
    result = report(files)
    assert result.exit_code == 1
    assert "unexpected RuntimeError: kaboom" in result.output
    assert "Traceback" not in result.output


def test_debug_prints_traceback(files, monkeypatch):
    seed(files["db"], [judged("cand", q) for q in ids()])

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "generate_report", boom)
    result = report(files, "--debug")
    assert result.exit_code == 1
    assert "Traceback" in result.output


def test_help_lists_the_options():
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    for option in ("--baseline", "--candidate", "--db", "--out", "--targets", "--passages",
                   "--splits-dir", "--top", "--open", "--debug"):
        assert option in result.output