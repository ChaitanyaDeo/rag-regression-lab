"""
tests/security/test_log_redaction.py

Day 11 (B6): secrets and e-mail addresses are redacted from log output.

  redact()           pure function: patterns, what must stay untouched, idempotence
  RedactingFilter    through a real logging handler: %-args, exception text, bad format args

Not covered on purpose (documented in docs/security.md): print() output bypasses logging;
stack_info text; phone numbers / IPs / user paths.

No network, no files, no models.

Run (repo root):  python -m pytest tests/security/test_log_redaction.py -v
"""

from __future__ import annotations

import io
import logging

import pytest

from ragregress.core.logging import REDACTED, RedactingFilter, redact

# Fake keys are assembled at runtime so this file never contains a literal secret shape
# (scripts/check_secrets.py scans every tracked file and all of git history).
SK = "sk" + "-abcdefghijklmnop1234567890"
GHP = "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789"
HF = "hf" + "_abcdefghijklmnopqrstuvwxyz0123456789"
CONFIG_HASH = "41c509687d97" + "0123456789abcdef" * 3 + "01234567"[:4]      # 64 hex chars


# ---------------------------------------------------------------- redact(): what is removed

def test_fake_email_becomes_redacted():
    assert redact("alice.smith+test@example.com") == REDACTED
    assert REDACTED == "[REDACTED]"


def test_email_inside_a_sentence_keeps_the_rest():
    assert redact("contact bob@example.org now, thanks") == "contact [REDACTED] now, thanks"


def test_every_email_in_the_text_is_redacted():
    out = redact("from a@x.io to b.c@sub.example.co.uk cc d_e@y.dev")
    assert out == "from [REDACTED] to [REDACTED] cc [REDACTED]"


@pytest.mark.parametrize("secret", [SK, GHP, HF], ids=["openai-style", "github", "huggingface"])
def test_api_key_shapes_are_redacted(secret):
    assert redact(f"using key {secret} for the call") == "using key [REDACTED] for the call"


def test_bearer_token_is_redacted_but_the_scheme_word_stays():
    assert redact("Authorization: Bearer abc.def-123_xyz==") == "Authorization: Bearer [REDACTED]"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("api_key=abc123", "api_key=[REDACTED]"),
        ("API-KEY: abc123", "API-KEY: [REDACTED]"),
        ("password: hunter2", "password: [REDACTED]"),
        ("passwd=hunter2, retrying", "passwd=[REDACTED], retrying"),
        ("auth_token=abc123 next", "auth_token=[REDACTED] next"),
        ('{"token": "abc 123", "n": 1}', '{"token": [REDACTED], "n": 1}'),
        ("secret='two words' ok", "secret=[REDACTED] ok"),
    ],
)
def test_key_value_secrets_keep_the_key_and_lose_the_value(text, expected):
    assert redact(text) == expected


# ---------------------------------------------------------------- redact(): what must stay

@pytest.mark.parametrize(
    "text",
    [
        "val-41c50968-20261023-145024",                                  # run id
        "config_hash=" + CONFIG_HASH,                                    # 64-hex hash
        "retrieved p0000042 and p0000007",                               # passage ids
        "[source: p0000042]",                                            # citation tag
        "Who formulated the law of gravitation?",
        "max_tokens=500 top_k=5 judge_seed=42",                          # `tokens` is not `token`
        "RuntimeError('judge exploded')",
        "decorator @property is fine",
        "D:\\Projects\\rag-regression-lab\\ragregress.db",
        "",
    ],
)
def test_ordinary_text_is_not_touched(text):
    assert redact(text) == text


@pytest.mark.parametrize(
    "text",
    [f"{SK} a@b.com", "api_key=abc123 Bearer tok.en password: x", "plain text", f"token={HF}"],
)
def test_redact_is_idempotent(text):
    once = redact(text)
    assert redact(once) == once


# ---------------------------------------------------------------- RedactingFilter through a real handler

@pytest.fixture
def log():
    """(logger, stream): a private logger whose only handler carries the filter."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger(f"test_log_redaction.{id(stream)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    yield logger, stream
    logger.removeHandler(handler)


def test_filter_redacts_the_message(log):
    logger, stream = log
    logger.info("login failed for carol@example.com")
    assert stream.getvalue() == "INFO login failed for [REDACTED]\n"


def test_filter_redacts_percent_style_args(log):
    logger, stream = log
    logger.warning("user %s used key %s on run %s", "dave@example.com", SK, "val-41c50968-20261023-145024")
    assert stream.getvalue() == "WARNING user [REDACTED] used key [REDACTED] on run val-41c50968-20261023-145024\n"


def test_filter_leaves_a_safe_message_unchanged(log):
    logger, stream = log
    logger.info("retrieved %d chunks for %s", 5, "p0000042")
    assert stream.getvalue() == "INFO retrieved 5 chunks for p0000042\n"


def test_filter_redacts_exception_text(log):
    logger, stream = log
    try:
        raise ValueError("bad password=hunter2 for erin@example.com")
    except ValueError:
        logger.exception("call failed")
    out = stream.getvalue()
    assert "call failed" in out and "ValueError" in out
    assert "hunter2" not in out and "erin@example.com" not in out
    assert "password=[REDACTED]" in out


def test_filter_never_drops_a_record():
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "x@y.com", None, None)
    assert RedactingFilter().filter(record) is True


def test_bad_format_args_do_not_raise_and_do_not_leak(log):
    logger, stream = log
    logger.info("two slots %s %s", "frank@example.com")          # one arg short
    out = stream.getvalue()
    assert "frank@example.com" not in out