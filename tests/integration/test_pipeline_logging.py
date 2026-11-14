"""
tests/integration/test_pipeline_logging.py

Day 13 (B1): the pipeline's structured logs through the REAL logging setup
(configure_logging + run_log_file), end to end through the compiled LangGraph graph.

  - one record per stage per query, all carrying the run_id; run-level stages too
  - retry attempts are numbered; exhausted retries end in one failed_query record
  - secrets inside exceptions are redacted on the console and in the per-run file
  - the pipeline prints nothing (stdout/stderr stay empty)
  - default console (WARNING) shows warnings only, while the file still has every INFO record
  - runner.execute writes artifacts/logs/<run_id>.jsonl next to the db; the CLI prints its path
  - the CLI reads LOG_LEVEL from .env in the working directory; a shell variable wins

Fake RAG system, real retrieval/F1 metrics, temp SQLite and temp log folder. No Ollama, no models.

Run (repo root):  python -m pytest tests/integration/test_pipeline_logging.py -v
"""

from __future__ import annotations

import io
import json
import logging
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragregress.benchmark.schemas import QAPair
from ragregress.core.logging import (
    APP_LOGGER, LOG_FIELDS, LOG_LEVEL_ENV, REDACTED, configure_logging, run_log_dir, run_log_file,
)
from ragregress.evaluation.metrics import MRR, RecallAtK, TokenPRF1
from ragregress.evaluation.metrics.base import MetricResult
from ragregress.pipeline.pipeline import MAX_RETRIES, Pipeline
from ragregress.rag.interface import Chunk, GenerationResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QA = [QAPair(**q) for q in json.loads((FIXTURES / "qa_5doc.json").read_text(encoding="utf-8"))]
BY_QUESTION = {q.question: q for q in QA}
Q0, Q1 = QA[0], QA[1]
RUN_ID = "val-logtest-20261028-120000"

# Fake secrets assembled at runtime (scripts/check_secrets.py scans tracked files).
SK = "sk" + "-abcdefghijklmnop1234567890"
EMAIL = "mallory" + "@example.com"
PASSWORD = "hunter" + "2secret"
SECRET_ERROR = f"auth failed for {EMAIL} password={PASSWORD} key {SK}"
SCHEMA_KEYS = {"ts", "level", "logger", "event", *LOG_FIELDS}


# ---------------------------------------------------------------- fakes

class FakeRag:
    """Gold passage at rank 1, first gold answer. fail_*: {question: n} -> first n calls raise."""

    def __init__(self, fail_retrieve=None, fail_generate=None, generate_error="generate boom", probe_fails=False):
        self.fail_retrieve, self.fail_generate = dict(fail_retrieve or {}), dict(fail_generate or {})
        self.generate_error, self.probe_fails = generate_error, probe_fails
        self.calls = Counter()
        self._lock = threading.Lock()

    def _maybe_fail(self, kind, question, budget, message):
        with self._lock:
            self.calls[(kind, question)] += 1
            n = self.calls[(kind, question)]
        if n <= budget.get(question, 0):
            raise RuntimeError(f"{message} #{n}")

    def retrieve(self, question, k=5):
        if question == "health check probe":
            if self.probe_fails:
                raise ConnectionError(f"vector store unreachable ({SECRET_ERROR})")
            return []
        self._maybe_fail("retrieve", question, self.fail_retrieve, "retrieve boom")
        gold = BY_QUESTION[question].gold_passage_ids[0]
        return [Chunk(chunk_id=f"{gold}::0", source_passage_id=gold, text="gold text")]

    def generate(self, question, contexts):
        self._maybe_fail("generate", question, self.fail_generate, self.generate_error)
        answer = BY_QUESTION[question].answer[0]
        return GenerationResult(raw_output=answer, answer=answer, citations=[])


class StubFaithfulness:
    name = "faithfulness"
    required_inputs = frozenset({"question", "answer", "context"})

    def __init__(self, crash_on=()):
        self.crash_on = set(crash_on)

    def score(self, *, question, answer, context):
        if question in self.crash_on:
            raise RuntimeError(f"judge exploded: {SECRET_ERROR}")
        return MetricResult(value=1.0)


def metrics(faith=None):
    return [RecallAtK(k=5), MRR(k=5), TokenPRF1(), faith or StubFaithfulness()]


# ---------------------------------------------------------------- helpers

@pytest.fixture(autouse=True)
def clean_logging(monkeypatch, tmp_path):
    monkeypatch.setenv(LOG_LEVEL_ENV, "unset")      # record the original value: the CLI's load_dotenv writes
    monkeypatch.delenv(LOG_LEVEL_ENV)               # os.environ, teardown must undo that
    monkeypatch.chdir(tmp_path)                     # the CLI loads .env from the cwd: never the repo's own .env
    root, app = logging.getLogger(), logging.getLogger(APP_LOGGER)
    saved = (list(root.handlers), root.level, app.level)
    yield
    for h in list(root.handlers):
        if h not in saved[0]:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved[1])
    app.setLevel(saved[2])


def run_logged(tmp_path, rag, metric_list=None, level="INFO", max_workers=1):
    """Pipeline.run with the real console setup + per-run file. Returns (state, console lines, file lines)."""
    console = io.StringIO()
    configure_logging(level=level, stream=console)
    db = tmp_path / "pipe.db"
    pipe = Pipeline(rag_system=rag, metrics=metric_list or metrics(), k=5, max_workers=max_workers, db_path=db)
    with run_log_file(RUN_ID, run_log_dir(db)) as path:
        state = pipe.run(qa_pairs=QA, run_id=RUN_ID, is_dev=False)
    file_text = path.read_text(encoding="utf-8")
    return state, console.getvalue(), file_text, path


def parse(text):
    return [json.loads(line) for line in text.splitlines()]


def by_stage(lines, stage, **match):
    return [l for l in lines if l["stage"] == stage and all(l[k] == v for k, v in match.items())]


# ---------------------------------------------------------------- happy path

def test_every_query_logs_each_stage_once_with_the_run_id(tmp_path):
    _, console, file_text, path = run_logged(tmp_path, FakeRag())
    lines = parse(file_text)

    assert path == tmp_path / "artifacts" / "logs" / f"{RUN_ID}.jsonl"
    assert {l["run_id"] for l in lines} == {RUN_ID}
    for stage in ("retrieve", "generate", "judge"):
        ok = by_stage(lines, stage, status="ok", level="INFO")
        assert sorted(l["query_id"] for l in ok) == sorted(q.qa_id for q in QA), stage
        assert all(isinstance(l["latency_ms"], (int, float)) and l["latency_ms"] >= 0 for l in ok)
    assert {l["attempt"] for l in by_stage(lines, "retrieve")} == {1}
    assert parse(console) == lines                      # console at INFO sees the same records


def test_run_level_stages_are_logged_in_order(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag())
    run_level = [(l["stage"], l["status"]) for l in parse(file_text) if l["query_id"] is None]
    # FakeRag has no ingest(): one "skipped" ingest record before the health probe
    assert run_level == [("run", "started"), ("ingest", "skipped"), ("ingest", "healthy"), ("aggregate", "ok"),
                         ("persist", "healthy"), ("run", "healthy")]


def test_every_line_has_exactly_the_schema_keys_and_null_tokens(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag())
    for line in parse(file_text):
        assert set(line) == SCHEMA_KEYS
        assert line["tokens_in"] is None and line["tokens_out"] is None


def test_persist_line_reports_rows_and_metrics(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag())
    (persist,) = by_stage(parse(file_text), "persist")
    assert "5 rows written" in persist["event"] and "'recall_at_k': 1.0" in persist["event"]


def test_pipeline_prints_nothing(tmp_path, capsys):
    run_logged(tmp_path, FakeRag(fail_retrieve={Q0.question: 99}), metrics(StubFaithfulness({Q1.question})))
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_parallel_run_tags_every_record(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag(fail_generate={Q1.question: 2}), max_workers=4)
    lines = parse(file_text)
    assert {l["run_id"] for l in lines} == {RUN_ID}
    assert len(by_stage(lines, "judge", status="ok")) == 5
    assert [l["attempt"] for l in by_stage(lines, "generate", query_id=Q1.qa_id)] == [1, 2, 3]


# ---------------------------------------------------------------- failures

def test_retry_attempts_are_numbered_and_recovery_is_ok(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag(fail_retrieve={Q1.question: 2}))
    attempts = [(l["attempt"], l["status"], l["level"]) for l in by_stage(parse(file_text), "retrieve",
                                                                          query_id=Q1.qa_id)]
    assert attempts == [(1, "failed", "WARNING"), (2, "failed", "WARNING"), (3, "ok", "INFO")]


def test_exhausted_retries_end_in_one_failed_query_record(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag(fail_retrieve={Q0.question: 99}))
    lines = parse(file_text)
    assert len(by_stage(lines, "retrieve", query_id=Q0.qa_id, status="failed")) == MAX_RETRIES
    (failed,) = by_stage(lines, "failed_query")
    assert (failed["query_id"], failed["level"], failed["status"]) == (Q0.qa_id, "ERROR", "failed")
    assert "retrieve boom #3" in failed["error"]
    assert by_stage(lines, "generate", query_id=Q0.qa_id) == []


def test_generate_failures_are_labelled_generate(tmp_path):
    # F-D13-1: the old print said "[retrieve]" for a generate failure
    _, _, file_text, _ = run_logged(tmp_path, FakeRag(fail_generate={Q0.question: 1}))
    (failed,) = by_stage(parse(file_text), "generate", status="failed")
    assert (failed["query_id"], failed["attempt"]) == (Q0.qa_id, 1)
    assert "generate boom #1" in failed["error"]


def test_metric_crash_is_a_warning_with_traceback_and_partial_judge(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag(), metrics(StubFaithfulness({Q0.question})))
    lines = parse(file_text)
    (crash,) = by_stage(lines, "judge", status="metric_failed")
    assert crash["level"] == "WARNING" and crash["query_id"] == Q0.qa_id
    assert crash["error"].startswith("faithfulness: RuntimeError")
    (partial,) = by_stage(lines, "judge", status="partial")
    assert partial["query_id"] == Q0.qa_id and "3/4 metrics scored" in partial["event"]


def test_unhealthy_run_logs_ingest_error_and_finishes_unhealthy(tmp_path):
    _, _, file_text, _ = run_logged(tmp_path, FakeRag(probe_fails=True))
    lines = parse(file_text)
    (ingest,) = by_stage(lines, "ingest", level="ERROR")
    assert ingest["status"] == "unhealthy" and "vector store unreachable" in ingest["error"]
    assert by_stage(lines, "retrieve") == []
    assert by_stage(lines, "run")[-1]["status"] == "unhealthy"


# ---------------------------------------------------------------- redaction end to end

def assert_no_secret(text):
    for secret in (SK, EMAIL, PASSWORD):
        assert secret not in text


def test_secrets_in_exceptions_are_redacted_in_console_and_file(tmp_path):
    rag = FakeRag(fail_generate={Q0.question: 99}, generate_error=SECRET_ERROR)
    _, console, file_text, _ = run_logged(tmp_path, rag, metrics(StubFaithfulness({Q1.question})))
    assert_no_secret(console)
    assert_no_secret(file_text)

    lines = parse(file_text)
    gen_fail = by_stage(lines, "generate", status="failed")[0]
    assert REDACTED in gen_fail["error"] and "password=[REDACTED]" in gen_fail["error"]
    (crash,) = by_stage(lines, "judge", status="metric_failed")
    assert "exc" in crash and REDACTED in crash["exc"]          # traceback kept, redacted


def test_warning_console_redacts_too(tmp_path):
    rag = FakeRag(fail_generate={Q0.question: 99}, generate_error=SECRET_ERROR)
    _, console, _, _ = run_logged(tmp_path, rag, level="WARNING")
    failed = [l for l in parse(console) if l["stage"] == "generate"]
    assert failed and all("password=[REDACTED]" in l["error"] for l in failed)
    assert_no_secret(console)


def test_unhealthy_detail_is_redacted_in_the_log(tmp_path):
    _, console, file_text, _ = run_logged(tmp_path, FakeRag(probe_fails=True))
    assert_no_secret(console + file_text)


# ---------------------------------------------------------------- console level vs file

def test_warning_console_shows_only_warnings_while_file_has_everything(tmp_path):
    _, console, file_text, _ = run_logged(tmp_path, FakeRag(fail_retrieve={Q1.question: 1}), level="WARNING")
    (only,) = parse(console)
    assert (only["level"], only["stage"], only["status"]) == ("WARNING", "retrieve", "failed")
    assert len(by_stage(parse(file_text), "retrieve", status="ok")) == 5


# ---------------------------------------------------------------- runner.execute and the CLI

def json_lines_in(output):
    """JSON log lines in CliRunner output (stdout and stderr are mixed, so a line may start with tqdm's bar)."""
    return [json.loads(line[line.index('{"'):]) for line in output.splitlines() if '{"' in line]


def fake_config(tmp_path):
    cfg = SimpleNamespace(
        pipeline=SimpleNamespace(id="fake", params=None), top_k=5, max_workers=1, is_dev=False,
        db_path=tmp_path / "run.db", eval_split="val", judge_provider="ollama", judge_model="fake", judge_seed=42,
    )
    cfg.split_path = lambda: tmp_path / "val.json"
    cfg.config_hash = lambda: "f" * 64
    cfg.model_dump = lambda mode="json": {"eval_split": "val", "top_k": 5}
    return cfg


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    import ragregress.core.runner as runner

    monkeypatch.setattr(runner, "load_qa_pairs", lambda path: QA)
    monkeypatch.setattr(runner, "build_rag_system", lambda pid, params: FakeRag(fail_retrieve={Q1.question: 1}))
    monkeypatch.setattr(runner, "build_metrics", lambda cfg, names=None: metrics())
    return runner, fake_config(tmp_path)


def test_execute_writes_the_run_log_next_to_the_db(tmp_path, fake_runner):
    runner, cfg = fake_runner
    outcome = runner.execute(cfg, run_id=RUN_ID)
    assert outcome.log_path == tmp_path / "artifacts" / "logs" / f"{RUN_ID}.jsonl"
    lines = parse(outcome.log_path.read_text(encoding="utf-8"))
    assert {l["run_id"] for l in lines} == {RUN_ID}
    assert len(by_stage(lines, "judge", status="ok")) == 5


def test_execute_keeps_the_log_of_an_unhealthy_run(tmp_path, fake_runner, monkeypatch):
    runner, cfg = fake_runner
    monkeypatch.setattr(runner, "build_rag_system", lambda pid, params: FakeRag(probe_fails=True))
    with pytest.raises(runner.RunFailedError):
        runner.execute(cfg, run_id=RUN_ID)
    lines = parse((tmp_path / "artifacts" / "logs" / f"{RUN_ID}.jsonl").read_text(encoding="utf-8"))
    assert by_stage(lines, "ingest", status="unhealthy")


def test_cli_run_prints_the_log_path_and_keeps_info_off_the_console(tmp_path, fake_runner, monkeypatch):
    from typer.testing import CliRunner
    import ragregress.cli as cli

    _, cfg = fake_runner
    monkeypatch.setattr(cli, "load_run_config", lambda path: cfg)
    result = CliRunner().invoke(cli.app, ["run", "--config", "x.yaml", "--run-id", RUN_ID])

    assert result.exit_code == 0, result.output
    log_path = tmp_path / "artifacts" / "logs" / f"{RUN_ID}.jsonl"
    assert f"log file    : {log_path}" in result.output
    console = json_lines_in(result.output)
    assert [(l["stage"], l["status"]) for l in console] == [("retrieve", "failed")]   # WARNING only
    assert len(by_stage(parse(log_path.read_text(encoding="utf-8")), "retrieve", status="ok")) == 5


def test_cli_run_with_info_level_prints_every_stage(tmp_path, fake_runner, monkeypatch):
    from typer.testing import CliRunner
    import ragregress.cli as cli

    _, cfg = fake_runner
    monkeypatch.setattr(cli, "load_run_config", lambda path: cfg)
    monkeypatch.setenv(LOG_LEVEL_ENV, "INFO")
    result = CliRunner().invoke(cli.app, ["run", "--config", "x.yaml", "--run-id", RUN_ID])

    assert result.exit_code == 0, result.output
    assert len(by_stage(json_lines_in(result.output), "judge", status="ok")) == 5


def test_cli_reads_log_level_from_dotenv_in_the_working_directory(tmp_path, fake_runner, monkeypatch):
    from typer.testing import CliRunner
    import os
    import ragregress.cli as cli

    _, cfg = fake_runner
    monkeypatch.setattr(cli, "load_run_config", lambda path: cfg)
    (tmp_path / ".env").write_text("LOG_LEVEL=INFO\n", encoding="utf-8")       # cwd is tmp_path (fixture)
    result = CliRunner().invoke(cli.app, ["run", "--config", "x.yaml", "--run-id", RUN_ID])

    assert result.exit_code == 0, result.output
    assert len(by_stage(json_lines_in(result.output), "judge", status="ok")) == 5
    assert os.environ[LOG_LEVEL_ENV] == "INFO"


def test_shell_log_level_wins_over_dotenv(tmp_path, fake_runner, monkeypatch):
    from typer.testing import CliRunner
    import ragregress.cli as cli

    _, cfg = fake_runner
    monkeypatch.setattr(cli, "load_run_config", lambda path: cfg)
    (tmp_path / ".env").write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    monkeypatch.setenv(LOG_LEVEL_ENV, "WARNING")
    result = CliRunner().invoke(cli.app, ["run", "--config", "x.yaml", "--run-id", RUN_ID])

    assert result.exit_code == 0, result.output
    assert [(l["stage"], l["status"]) for l in json_lines_in(result.output)] == [("retrieve", "failed")]