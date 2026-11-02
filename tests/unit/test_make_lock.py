"""
tests/unit/test_make_lock.py

Day 13 B4: scripts/make_lock.py (requirements.lock) and, when the lock file exists, that it agrees with
requirements.txt. No Docker, no pip: the script's text handling is a pure function.

Run (repo root):  python -m pytest tests/unit/test_make_lock.py -v
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_lock.py"
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock"


@pytest.fixture(scope="module")
def make_lock():
    spec = importlib.util.spec_from_file_location("make_lock", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = """\
annotated-types==0.7.0
fastapi==0.141.1
ragregress @ file:///app
Torch==2.11.0+cpu
pydantic==2.12.5
"""


def test_the_package_itself_is_left_out(make_lock):
    assert not [l for l in make_lock.lock_lines(FREEZE) if l.lower().startswith("ragregress")]


def test_the_local_version_tag_is_dropped_but_the_version_stays(make_lock):
    lines = make_lock.lock_lines(FREEZE)
    assert "Torch==2.11.0" in lines
    assert not [l for l in lines if "+" in l]


def test_lines_are_sorted_by_normalised_name_and_the_text_ends_with_a_newline(make_lock):
    lines = make_lock.lock_lines(FREEZE)
    assert lines == ["annotated-types==0.7.0", "fastapi==0.141.1", "pydantic==2.12.5", "Torch==2.11.0"]
    text = make_lock.lock_text(FREEZE)
    assert text.startswith("# requirements.lock") and text.endswith("\n") and "\r" not in text


def test_every_package_line_is_an_exact_pin(make_lock):
    for line in make_lock.lock_lines(FREEZE):
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[^=\s]+", line), line


def test_comments_blank_and_editable_lines_are_ignored(make_lock):
    assert make_lock.lock_lines("# x\n\n-e git+https://example/x#egg=x\nabc==1\n") == ["abc==1"]


def _direct_pins() -> dict:
    pins = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?)==([^\s#;]+)", raw.strip())
        if match:
            pins[re.sub(r"\[.*\]", "", match.group(1))] = match.group(2)
    return pins


@pytest.mark.skipif(not LOCK.is_file(), reason="requirements.lock not generated yet")
def test_the_lock_file_contains_every_direct_pin_at_the_same_version(make_lock):
    locked = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            locked[make_lock.normalise(name)] = version
    for name, version in _direct_pins().items():
        assert locked.get(make_lock.normalise(name)) == version, f"{name}=={version} is not in requirements.lock"


@pytest.mark.skipif(not LOCK.is_file(), reason="requirements.lock not generated yet")
def test_the_lock_file_is_plain_pins_with_no_local_tags_and_no_ragregress():
    body = [l for l in LOCK.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    assert body and all("==" in l and "+" not in l for l in body)
    assert not [l for l in body if l.lower().startswith("ragregress")]