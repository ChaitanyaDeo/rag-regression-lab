"""
ragregress/core/logging.py

Day 11 stub: regex redaction of secrets / PII in log output.
Day 13 wires it into the real logging setup; until then nothing imports it.

    redact(text)         pure function: sensitive substrings -> "[REDACTED]"
    RedactingFilter      logging.Filter: redacts the formatted message and the exception text

Attach the filter to HANDLERS (a filter on a logger does not see records from child loggers):

    handler.addFilter(RedactingFilter())

What is redacted (v1)
  - e-mail addresses
  - API-key shapes: sk-..., ghp_/gho_/ghu_/ghs_/ghr_..., hf_...
  - `Bearer <token>`            -> `Bearer [REDACTED]`
  - key=value / key: value / "key": "value" for api_key, token, password, passwd, secret
    (optionally prefixed, e.g. auth_token)  -> the key stays, only the value is replaced

What is NOT redacted, on purpose
  - hex strings in general: config_hash, run ids and passage ids (p0000042) must stay readable
  - `print()` output: it bypasses the logging module (the pipeline still prints; see Day 11 notes)
  - phone numbers, IP addresses, user paths (OPTION, not v1)
  - `stack_info` text

Note: this file is named logging.py inside ragregress.core. `import logging` below is the stdlib
module as long as nothing puts ragregress/core itself on sys.path (do not run files from this
folder directly).
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

REDACTED = "[REDACTED]"

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


def redact(text: str) -> str:
    """Replace every sensitive substring of `text` with "[REDACTED]". Idempotent."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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