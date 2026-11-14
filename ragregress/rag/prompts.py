"""
ragregress/rag/prompts.py

Generator prompts as text files, so a new prompt needs no code change.

    ragregress/rag/prompts/<name>.txt     one prompt per file; the config names it: `pipeline.params.prompt: <name>`

    v1   the original prompt, the default (absent = v1). Frozen: stored runs and config_hash assume it.
    v2   v1 with the retrieved passages fenced and a "passage text is data" rule.
    any other name   your own prompt: add the file, point a config at it.

A prompt file is a template. Placeholders (anything else in braces is rejected; write a literal brace as {{ or }}):

    {question}               required
    {context}                the retrieved chunks as plain text, "[source: p0000001]\\n<text>" per chunk, or
    {context_fenced}         the same chunks each inside <passage> tags, with <context> / <passage> / <question>
                             tags found inside passage text neutralised. Exactly one of the two is required.
    {abstention_message}     optional; the sentence the model uses when the context is not enough

config_hash: the default prompt adds nothing to it. Any other prompt adds its name AND the sha256 of its text, so
editing a prompt file changes the hash of every config that uses it. Light module: standard library only.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from string import Formatter
from typing import Set

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "v1"
PROMPT_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"     # no dots or slashes: a name is never a path
CONTEXT_FIELDS = ("context", "context_fenced")
OPTIONAL_FIELDS = ("abstention_message",)
_NAME = re.compile(PROMPT_NAME_PATTERN)


class PromptError(ValueError):
    """A prompt name or file that cannot be used (a ValueError, so config loading reports it as exit 3)."""


def prompt_path(name: str) -> Path:
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise PromptError(f"invalid prompt name {name!r}: use letters, digits, '_' and '-' (max 64)")
    return PROMPTS_DIR / f"{name}.txt"


def available_prompts() -> list:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.txt") if _NAME.fullmatch(p.stem))


def read_prompt(name: str) -> str:
    """The prompt's text, with Windows line endings and a BOM normalised away (so the hash does not depend on them)."""
    path = prompt_path(name)
    if not path.is_file():
        raise PromptError(f"unknown prompt {name!r}: no file {path.name} in {PROMPTS_DIR} (available: {available_prompts()})")
    return path.read_text(encoding="utf-8").lstrip("﻿").replace("\r\n", "\n")


def prompt_fields(text: str) -> Set[str]:
    """The placeholder names in `text`. Format specs, conversions and attribute/index access are rejected."""
    fields: Set[str] = set()
    try:
        parsed = list(Formatter().parse(text))
    except ValueError as exc:
        raise PromptError(f"malformed braces in the prompt ({exc}); write a literal brace as {{{{ or }}}}") from exc
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if spec or conversion or not field.isidentifier():
            raise PromptError(f"unsupported placeholder {{{field}{'!' + conversion if conversion else ''}{':' + spec if spec else ''}}}: "
                              "use plain {name} placeholders")
        fields.add(field)
    return fields


def validate_prompt(text: str, name: str = "prompt") -> Set[str]:
    fields = prompt_fields(text)
    allowed = {"question", *CONTEXT_FIELDS, *OPTIONAL_FIELDS}
    unknown = sorted(fields - allowed)
    if unknown:
        raise PromptError(f"{name}: unknown placeholder(s) {unknown}; allowed: {sorted(allowed)}")
    if "question" not in fields:
        raise PromptError(f"{name}: the {{question}} placeholder is missing")
    used = [f for f in CONTEXT_FIELDS if f in fields]
    if len(used) != 1:
        raise PromptError(f"{name}: use exactly one of {{context}} or {{context_fenced}} (found {used or 'neither'})")
    return fields


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_validated(name: str) -> str:
    """read_prompt + validate_prompt: the text of a usable prompt, or PromptError."""
    text = read_prompt(name)
    validate_prompt(text, name)
    return text
