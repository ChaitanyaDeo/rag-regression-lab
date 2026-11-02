"""
tests/unit/test_logging.py

logging setup in ragregress/core/logging.py (stdlib logging + python-json-logger).

  JsonFormatter      fixed keys, nulls for unset fields, every string value redacted, valid JSON
  configure_logging  JSON console handler: level from LOG_LEVEL, DEBUG reaches the console, idempotent,
                     follows sys.stderr
  run_log_file       per-run JSON file at INFO while the console stays at WARNING; safe file names

No network, no models. Files go to tmp_path. Logging state is restored after each test.

Run (repo root):  python -m pytest tests/unit/test_logging.py -v
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import pytest

from ragregress.core.logging import (
    APP_LOGGER, LOG_FIELDS, LOG_LEVEL_ENV, REDACTED, JsonFormatter, RedactingFilter, configure_logging,
    run_log_dir, run_log_file, run_log_path,
)

# Fake secrets are assembled at runtime (scripts/check_secrets.py scans tracked files).
SK = "sk" + "-abcdefghijklmnop1234567890"
EMAIL = "erin" + "@example.com"
RUN_ID = "val-41c50968-20261023-145024"
SCHEMA_KEYS = {"ts", "level", "logger", "event", *LOG_FIELDS}


@pytest.fixture(autouse=True)
def clean_logging(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "unset")              # record the original value so teardown restores it,
    monkeypatch.delenv(LOG_LEVEL_ENV)                       # even if code under test writes os.environ
    root, app = logging.getLogger(), logging.getLogger(APP_LOGGER)
    saved = (list(root.handlers), root.level, app.level)
    yield
    for h in list(root.handlers):
        if h not in saved[0]:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved[1])
    app.setLevel(saved[2])


def fmt(msg="hello", level=logging.INFO, exc_info=None, **extra) -> dict:
    record = logging.LogRecord("ragregress.test", level, __file__, 1, msg, None, exc_info)
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def lines(text: str):
    return [json.loads(line) for line in text.splitlines()]


# ---------------------------------------------------------------- JsonFormatter

def test_every_schema_key_is_present_and_unset_fields_are_null():
    out = fmt("retrieve ok", run_id=RUN_ID, stage="retrieve", attempt=1, latency_ms=3.2)
    assert set(out) == SCHEMA_KEYS
    assert (out["level"], out["logger"], out["event"]) == ("INFO", "ragregress.test", "retrieve ok")
    assert (out["run_id"], out["stage"], out["attempt"], out["latency_ms"]) == (RUN_ID, "retrieve", 1, 3.2)
    assert out["query_id"] is None and out["tokens_in"] is None and out["error"] is None
    assert out["ts"].endswith("+00:00")


def test_secret_in_an_extra_field_is_redacted():
    out = fmt(error=f"RuntimeError('auth failed for {EMAIL} key {SK}')")
    assert out["error"] == f"RuntimeError('auth failed for {REDACTED} key {REDACTED}')"


def test_quoted_secret_value_keeps_the_line_valid_json():
    out = fmt('payload {"token": "abc 123"}', error='{"password": "x y"}')     # fmt() parses the JSON
    assert out["event"] == 'payload {"token": [REDACTED]}'
    assert out["error"] == '{"password": [REDACTED]}'


def test_ids_and_hashes_are_not_redacted():
    out = fmt("retrieved p0000042", run_id=RUN_ID, query_id="57112686b654c5140001fbd7")
    assert (out["event"], out["run_id"], out["query_id"]) == ("retrieved p0000042", RUN_ID,
                                                               "57112686b654c5140001fbd7")


def test_exception_text_is_included_as_exc_and_redacted():
    try:
        raise ValueError(f"bad password=hunter2 for {EMAIL}")
    except ValueError:
        out = fmt("judge failed", exc_info=sys.exc_info())
    assert "ValueError" in out["exc"] and "password=[REDACTED]" in out["exc"]
    assert "hunter2" not in out["exc"] and EMAIL not in out["exc"]


# ---------------------------------------------------------------- configure_logging

def our_handlers():
    return [h for h in logging.getLogger().handlers if getattr(h, "_ragregress_console", False)]


def test_default_console_is_json_at_warning():
    handler = configure_logging(stream=io.StringIO())
    assert isinstance(handler.formatter, JsonFormatter)
    assert any(isinstance(f, RedactingFilter) for f in handler.filters)
    assert handler.level == logging.WARNING
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger(APP_LOGGER).level == logging.INFO


def test_level_from_env_and_argument(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    assert configure_logging(stream=io.StringIO()).level == logging.DEBUG
    assert configure_logging(level="ERROR", stream=io.StringIO()).level == logging.ERROR   # argument wins


def test_env_variable_is_named_log_level():
    assert LOG_LEVEL_ENV == "LOG_LEVEL"                     # the name in .env.example and the plan


def test_debug_level_lets_debug_records_through(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "DEBUG")
    stream = io.StringIO()
    configure_logging(stream=stream)
    assert logging.getLogger(APP_LOGGER).level == logging.DEBUG
    logging.getLogger("ragregress.x").debug("detail")
    logging.getLogger("httpx").debug("third-party detail")   # root stays at WARNING
    assert [l["event"] for l in lines(stream.getvalue())] == ["detail"]


def test_error_level_keeps_the_app_logger_at_info():
    configure_logging(level="ERROR", stream=io.StringIO())
    assert logging.getLogger(APP_LOGGER).level == logging.INFO     # the run file still gets INFO


def test_invalid_env_level_falls_back_to_warning_and_says_so(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "LOUD")
    stream = io.StringIO()
    assert configure_logging(stream=stream).level == logging.WARNING
    assert "unknown LOG_LEVEL='LOUD'" in stream.getvalue()


def test_second_call_replaces_the_first_handler():
    first = configure_logging(stream=io.StringIO())
    second = configure_logging(stream=io.StringIO())
    assert our_handlers() == [second] and first not in logging.getLogger().handlers


def test_warning_console_drops_info():
    stream = io.StringIO()
    configure_logging(stream=stream)
    log = logging.getLogger("ragregress.x")
    log.info("hidden")
    log.warning("shown")
    assert [l["event"] for l in lines(stream.getvalue())] == ["shown"]


def test_at_info_our_records_pass_and_third_party_info_does_not():
    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream)
    logging.getLogger("ragregress.pipeline.pipeline").info("ours")
    logging.getLogger("httpx").info("HTTP Request: POST http://localhost:11434")
    logging.getLogger("httpx").warning("theirs, but a warning")
    assert [l["event"] for l in lines(stream.getvalue())] == ["ours", "theirs, but a warning"]


def test_default_stream_follows_sys_stderr_swapped_after_setup(monkeypatch):
    configure_logging()
    swapped = io.StringIO()
    monkeypatch.setattr(sys, "stderr", swapped)             # CliRunner and the CLI's tqdm bar do this
    logging.getLogger("ragregress.x").warning("key %s", SK)
    (line,) = lines(swapped.getvalue())
    assert line["event"] == f"key {REDACTED}"


# ---------------------------------------------------------------- run log file

def test_run_log_dir_is_artifacts_logs_next_to_the_db(tmp_path):
    assert run_log_dir(tmp_path / "test.db") == tmp_path / "artifacts" / "logs"
    assert run_log_dir("ragregress.db") == Path("artifacts") / "logs"


@pytest.mark.parametrize("run_id, name", [
    (RUN_ID, f"{RUN_ID}.jsonl"),
    ("../../etc/passwd", "_.._etc_passwd.jsonl"),
    ("a b/c\\d", "a_b_c_d.jsonl"),
    ("..", "_.jsonl"),
])
def test_run_log_path_is_a_safe_file_name(tmp_path, run_id, name):
    path = run_log_path(run_id, tmp_path)
    assert path.parent == tmp_path and path.name == name


def test_file_gets_info_while_console_stays_at_warning(tmp_path):
    console = io.StringIO()
    configure_logging(stream=console)
    log = logging.getLogger("ragregress.pipeline.pipeline")
    with run_log_file(RUN_ID, tmp_path) as path:
        log.info("retrieve ok", extra={"run_id": RUN_ID})
        log.warning("retrieve failed", extra={"run_id": RUN_ID, "error": SK})
    file_lines = lines(path.read_text(encoding="utf-8"))
    assert [(l["level"], l["event"]) for l in file_lines] == [("INFO", "retrieve ok"), ("WARNING", "retrieve failed")]
    assert file_lines[1]["error"] == REDACTED
    assert [l["event"] for l in lines(console.getvalue())] == ["retrieve failed"]
    assert path == tmp_path / f"{RUN_ID}.jsonl"


def test_file_gets_info_without_configure_logging(tmp_path):
    with run_log_file(RUN_ID, tmp_path) as path:
        logging.getLogger("ragregress.x").info("no console set up")
    assert [l["event"] for l in lines(path.read_text(encoding="utf-8"))] == ["no console set up"]


def test_file_handler_is_removed_afterwards_even_on_error(tmp_path):
    before = list(logging.getLogger().handlers)
    with pytest.raises(RuntimeError):
        with run_log_file(RUN_ID, tmp_path):
            raise RuntimeError("run crashed")
    assert logging.getLogger().handlers == before


def test_same_run_id_appends(tmp_path):
    log = logging.getLogger("ragregress.x")
    for event in ("first", "second"):
        with run_log_file(RUN_ID, tmp_path) as path:
            log.info(event)
    assert [l["event"] for l in lines(path.read_text(encoding="utf-8"))] == ["first", "second"]