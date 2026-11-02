"""
tests/security/test_secrets_exposure.py

Day 11 (B7): guard tests on the REAL repository. They fail the suite if someone later
  - removes `.env` (or a variant) from .gitignore, or ignores `.env.example`,
  - puts a real-looking value in `.env.example`,
  - tracks an .env file, or commits a secret shape into any tracked file.

Read-only: runs `git ls-files` / `git check-ignore` and reads tracked files. The full-history scan
is NOT part of the suite (it grows with the repo): run it by hand and record the result:

    python scripts/check_secrets.py

Skipped when git is missing or the repo root is not a git work tree (e.g. a source archive).

Run (repo root):  python -m pytest tests/security/test_secrets_exposure.py -v
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_secrets.py"


def _in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
                          capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


pytestmark = pytest.mark.skipif(not _in_git_work_tree(), reason="not a git work tree (or git missing)")


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("check_secrets_real_repo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                      # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


def show(findings):
    return "\n".join(f"  {f.check}: {f}" for f in findings)


def test_env_files_are_ignored_and_the_example_is_not(cs):
    findings = cs.check_gitignore(ROOT)
    assert findings == [], "\n" + show(findings)


def test_env_example_holds_placeholders_only(cs):
    findings = cs.check_env_example(ROOT)
    assert findings == [], "\n" + show(findings)


def test_no_env_file_and_no_secret_shape_is_tracked(cs):
    findings = cs.scan_tracked_files(ROOT)
    assert findings == [], "\n" + show(findings)