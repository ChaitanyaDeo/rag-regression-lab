"""
ragregress/core/logging.py

Logging for RAG Regression Lab: stdlib `logging` + python-json-logger, with redaction.

    redact(text)            pure function: sensitive substrings -> "[REDACTED]"
    RedactingFilter         logging.Filter: redacts the formatted message and the exception text
    JsonFormatter           python-json-logger formatter: fixed keys, every string value redacted
    configure_logging()     console handler (JSON, stderr) for entry points (CLI / API)
    run_log_file()          per-run JSON file <db folder>/artifacts/logs/<run_id>.jsonl, INFO and up

Library modules only call logging.getLogger(__name__) and never add handlers.

Every JSON line has: ts, level, logger, event, run_id, stage, query_id, attempt, status,
latency_ms, tokens_in, tokens_out, error (null when not set), plus "exc" when an exception is attached.
Pipeline code passes the schema fields with `extra={...}`.

Levels: each handler chooses what it shows.
    console  LOG_LEVEL (DEBUG / INFO / WARNING / ERROR; default WARNING). Set it in the shell or in
           .env (the CLI loads .env from the working directory; a shell variable wins over .env).
  file     INFO, always

The `ragregress` logger is at INFO, or at DEBUG when LOG_LEVEL=DEBUG
Third-party loggers stay at the root level (WARNING).

Assumes one run at a time per process: the run file takes every ragregress record while it is open.

What is redacted (v1)
  - e-mail addresses
  - API-key shapes: sk-..., ghp_/gho_/ghu_/ghs_/ghr_..., hf_...
  - `Bearer <token>`            -> `Bearer [REDACTED]`
  - key=value / key: value / "key": "value" for api_key, token, password, passwd, secret
    (optionally prefixed, e.g. auth_token)  -> the key stays, only the value is replaced

JsonFormatter redacts each value before serialising: redacting the finished JSON line would
break it (`"token": "x"` -> `"token": [REDACTED]`).

What is NOT redacted, on purpose
  - hex strings in general: config_hash, run ids and passage ids (p0000042) must stay readable
  - phone numbers, IP addresses, user paths (OPTION, not v1)
  - `stack_info` text
  - output that does not go through logging (print, typer.echo, tqdm)

Note: this file is named logging.py inside ragregress.core. `import logging` below is the stdlib
module as long as nothing puts ragregress/core itself on sys.path (do not run files from this
folder directly).
"""

from __future__ import annotations
 
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, TextIO, Tuple, Union

from pythonjsonlogger.json import JsonFormatter as _LibJsonFormatter

REDACTED = "[REDACTED]"
 
LOG_LEVEL_ENV = "LOG_LEVEL"            # documented in .env.example and the plan (§3)
APP_LOGGER = "ragregress"
RUN_LOG_SUBDIR = Path("artifacts") / "logs"

# Schema fields carried by `extra={...}` 
LOG_FIELDS = ("run_id", "stage", "query_id", "attempt", "status",
              "latency_ms", "tokens_in", "tokens_out", "error")

_VALUE = r"""("[^"]*"|'[^']*'|[^\s,;]+)"""
_KEY_NAMES = r"(?:api[_-]?key|token|password|passwd|secret)"

# (compiled pattern, replacement). Order matters: Bearer first, e-mail last.
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE), rf"\1{REDACTED}"),
    (re.compile(rf"""\b((?:[A-Za-z0-9]+[_-])?{_KEY_NAMES})(["']?\s*[=:]\s*){_VALUE}""", re.IGNORECASE),
     rf"\1\2{REDACTED}"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), REDACTED),
]


# ---------------------------------------------------------------- redaction

def redact(text: str) -> str:
    """Replace every sensitive substring of `text` with "[REDACTED]". Idempotent."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def redact_value(value: Any) -> Any:
    """redact() every string inside `value` (dicts, lists, tuples recursively). Keys are left alone."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    return value


class RedactingFilter(logging.Filter):
    """Redacts record.msg (after %-formatting) and the exception text. Never drops a record."""
 
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:                      # bad format args: never crash the caller, never leak the args
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
 
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


# ---------------------------------------------------------------- formatters

class JsonFormatter(_LibJsonFormatter):
    """python-json-logger with the project schema; every string value is redacted before json.dumps."""
 
    def __init__(self) -> None:
        super().__init__(
            "%(levelname)s %(name)s %(message)s",
            rename_fields={"levelname": "level", "name": "logger", "message": "event", "exc_info": "exc"},
            defaults=dict.fromkeys(LOG_FIELDS),
            timestamp="ts",
            json_ensure_ascii=False,
        )
 
    def process_log_record(self, log_data: dict) -> dict:
        return redact_value(log_data)

# ---------------------------------------------------------------- handlers

class _CurrentStderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time (CliRunner and the CLI's tqdm bar replace it)."""
 
    def __init__(self) -> None:
        super().__init__(sys.stderr)
 
    @property
    def stream(self):                          # type: ignore[override]
        return sys.stderr
 
    @stream.setter
    def stream(self, value) -> None:           # StreamHandler.__init__ assigns it; ignored
        pass


def _new_handler(handler: logging.Handler, level: int) -> logging.Handler:
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())
    handler.setLevel(level)
    return handler

def configure_logging(level: Union[str, int, None] = None, stream: Optional[TextIO] = None) -> logging.Handler:
    """Install (or replace) the JSON console handler on the root logger. Call from entry points only.
 
    level defaults to the LOG_LEVEL environment variable, else WARNING. stream=None means the
    current sys.stderr. Loading .env is the entry point's job (see cli.py), not this function's.
    """

    raw = level if level is not None else os.environ.get(LOG_LEVEL_ENV)

    resolved = None

    if raw in (None, ""):
        resolved = logging.WARNING
    elif isinstance(raw, int):
        resolved = raw
    else: 
        resolved = logging.getLevelName(str(raw).strip().upper())

    problem = None
    if not isinstance(resolved, int):
        problem, resolved = f"unknown {LOG_LEVEL_ENV}={raw!r}, using WARNING", logging.WARNING

    root = logging.getLogger()

    for old in [h for h in root.handlers if getattr(h, "_ragregress_console", False)]:
        root.removeHandler(old)
        old.close()

    handler = _new_handler(logging.StreamHandler(stream) if stream is not None else _CurrentStderrHandler(),
                           resolved)

    handler._ragregress_console = True         # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger(APP_LOGGER).setLevel(min(logging.INFO, resolved))   # DEBUG reaches the console

    if problem:
        logging.getLogger(__name__).warning("logging setup: %s", problem)
    return handler

def run_log_dir(db_path: Union[str, Path]) -> Path:
    """artifacts/logs next to the database: ragregress.db -> artifacts/logs; tmp/test.db -> tmp/artifacts/logs."""
    return Path(db_path).parent / RUN_LOG_SUBDIR


def run_log_path(run_id: str, log_dir: Union[str, Path]) -> Path:
    """<log_dir>/<run_id>.jsonl; characters outside [A-Za-z0-9._-] become '_' (no path traversal)."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", run_id).lstrip(".") or "_"
    return Path(log_dir) / f"{name}.jsonl"

@contextmanager
def run_log_file(run_id: str, log_dir: Union[str, Path]) -> Iterator[Path]:
    """Append INFO+ records to <log_dir>/<run_id>.jsonl (JSON, redacted) while the block runs."""
    path = run_log_path(run_id, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
 
    app = logging.getLogger(APP_LOGGER)
    if app.getEffectiveLevel() > logging.INFO:  # configure_logging() not called (tests, library use)
        app.setLevel(logging.INFO)
 
    handler = _new_handler(logging.FileHandler(path, mode="a", encoding="utf-8"), logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()