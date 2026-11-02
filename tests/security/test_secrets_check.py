"""
tests/security/test_secrets_check.py

Day 11 (B7): the logic of scripts/check_secrets.py, exercised on throw-away git repositories
(tmp_path). Nothing here touches the real repo, its history or any real secret: every
"secret" is a fake assembled at runtime (so this file itself contains no secret shape).

Covers: .gitignore rules, .env.example placeholders, secret shapes in tracked files, a secret
that was committed and later deleted (history), an .env file committed once, masked output,
exit codes of main().

Skipped when `git` is not installed.

Run (repo root):  python -m pytest tests/security/test_secrets_check.py -v
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_secrets.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _load_script():
    spec = importlib.util.spec_from_file_location("check_secrets_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                      # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


cs = _load_script()

# fake secrets, assembled so no literal shape appears in this file
FAKE_SK = "sk" + "-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
FAKE_GHP = "ghp" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FAKE_HF = "hf" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FAKE_AWS = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
FAKE_PEM = "-----BEGIN " + "RSA PRIVATE KEY-----"

GOOD_GITIGNORE = ".env\n.env.*\n!.env.example\n*.db\n"          # D's rules after the Q14 change (subset)
GOOD_EXAMPLE = "OLLAMA_HOST=http://localhost:11434\nLOG_LEVEL=INFO\n"


# ---------------------------------------------------------------- throw-away repos

def git(repo, *args):
    """Run git in a throw-away repo, immune to the developer's global config (signing, hooks)."""
    args = list(args)
    if args and args[0] == "commit":
        args.insert(1, "--no-verify")                    # skip any global pre-commit / commit-msg hook
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    return root


def commit(repo, files: dict, message="c"):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", "-A", "-f")
    git(repo, "commit", "-q", "-m", message)


def clean_repo(repo):
    commit(repo, {".gitignore": GOOD_GITIGNORE, ".env.example": GOOD_EXAMPLE, "src/app.py": "print('hi')\n"})
    return repo


def details(findings):
    return [(f.check, f.where) for f in findings]


# ---------------------------------------------------------------- clean repo

def test_clean_repo_has_no_findings(repo):
    assert cs.run_all(clean_repo(repo)) == []


def test_main_returns_0_and_prints_pass_for_a_clean_repo(repo, capsys):
    clean_repo(repo)
    assert cs.main(["--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out and "[FAIL]" not in out


# ---------------------------------------------------------------- .gitignore

def test_missing_env_rule_is_a_finding(repo):
    commit(repo, {".gitignore": "*.db\n", ".env.example": GOOD_EXAMPLE})
    got = {f.where for f in cs.check_gitignore(repo)}
    assert got == {".env", ".env.local", ".env.production", ".env.development"}


def test_env_variants_not_covered_are_findings(repo):
    commit(repo, {".gitignore": ".env\n", ".env.example": GOOD_EXAMPLE})      # today's rules, before Q14
    got = {f.where for f in cs.check_gitignore(repo)}
    assert got == {".env.local", ".env.production", ".env.development"}


def test_ignoring_the_example_file_is_a_finding(repo):
    commit(repo, {".gitignore": ".env*\n", ".env.example": GOOD_EXAMPLE})
    findings = cs.check_gitignore(repo)
    assert [(f.where, "must stay trackable" in f.detail) for f in findings] == [(".env.example", True)]


def test_good_gitignore_passes(repo):
    commit(repo, {".gitignore": GOOD_GITIGNORE})
    assert cs.check_gitignore(repo) == []


# ---------------------------------------------------------------- .env.example

@pytest.mark.parametrize(
    "line",
    [
        "OLLAMA_HOST=http://localhost:11434", "LOG_LEVEL=INFO", "LOG_LEVEL=debug", "API_KEY=", "API_KEY=<your-key>",
        "API_KEY=changeme", "API_KEY=your-api-key-here", "API_KEY=xxxxxxxx", "API_KEY=ollama", "PORT=8000",
        "BASE=https://example.com/v1", "export LOG_LEVEL=INFO", "# HF_TOKEN=whatever", "", "KEY=${OTHER}",
    ],
)
def test_placeholder_lines_in_env_example_pass(repo, line):
    commit(repo, {".env.example": line + "\n"})
    assert cs.check_env_example(repo) == []


@pytest.mark.parametrize(
    "line",
    [f"HF_TOKEN={FAKE_HF}", f"OPENAI_API_KEY={FAKE_SK}", "OPENAI_API_KEY=a1b2c3d4e5f6", "DB_URL=postgres://user:pw@host/db",
     "OLLAMA_HOST=http://user:pw@host:11434"],
    ids=["hf-shape", "openai-shape", "real-looking-value", "url-with-password", "host-with-userinfo"],
)
def test_real_looking_lines_in_env_example_are_findings(repo, line):
    commit(repo, {".env.example": line + "\n"})
    findings = cs.check_env_example(repo)
    assert findings and all(f.check == "env-example" and f.where.startswith(".env.example:1") for f in findings)


def test_missing_env_example_is_a_finding(repo):
    commit(repo, {"a.txt": "x\n"})
    assert details(cs.check_env_example(repo)) == [("env-example", ".env.example")]


# ---------------------------------------------------------------- tracked files

@pytest.mark.parametrize(
    "secret, label",
    [(FAKE_SK, "openai-style key"), (FAKE_GHP, "github token"), (FAKE_HF, "huggingface token"),
     (FAKE_AWS, "aws access key id"), (FAKE_PEM, "private key block")],
)
def test_secret_shapes_in_tracked_files_are_found_with_path_and_line(repo, secret, label):
    commit(repo, {"src/config.py": f"# header\nTOKEN = '{secret}'\n"})
    findings = cs.scan_tracked_files(repo)
    assert [(f.where, label in f.detail) for f in findings] == [("src/config.py:2", True)]


def test_ordinary_text_in_tracked_files_is_not_flagged(repo):
    commit(repo, {"README.md": "risk-adjusted task-based evaluation; sk-learn; ghp is not a token; hf_ prefix\n",
                  "data.json": '{"passage": "p0000042", "hash": "41c509687d97"}\n'})
    assert cs.scan_tracked_files(repo) == []


def test_tracked_env_file_is_a_finding_even_without_a_secret_shape(repo):
    commit(repo, {".env": "SOMETHING=1\n", ".env.example": GOOD_EXAMPLE})      # force-added past any ignore
    assert details(cs.scan_tracked_files(repo)) == [("tracked", ".env")]


def test_binary_files_are_skipped(repo):
    (repo / "blob.bin").write_bytes(b"\0\1\2" + FAKE_SK.encode())
    commit(repo, {"a.txt": "x\n"})
    assert cs.scan_tracked_files(repo) == []


# ---------------------------------------------------------------- history

def test_secret_committed_then_deleted_is_found_in_history_only(repo):
    clean_repo(repo)
    commit(repo, {"src/leak.py": f"KEY = '{FAKE_SK}'\n"}, message="oops")
    git(repo, "rm", "-q", "src/leak.py")
    git(repo, "commit", "-q", "-m", "remove it")

    assert cs.scan_tracked_files(repo) == []                      # gone from the working tree ...
    findings = cs.scan_history(repo)                              # ... but not from history
    assert [(f.where.split(":", 1)[1], "openai-style key" in f.detail) for f in findings] == [("src/leak.py", True)]


def test_history_scan_covers_other_branches(repo):
    clean_repo(repo)
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, {"side.py": f"K = '{FAKE_GHP}'\n"})
    git(repo, "checkout", "-q", "-")
    assert any("side.py" in f.where for f in cs.scan_history(repo))


def test_env_file_committed_once_is_found_even_after_removal(repo):
    clean_repo(repo)
    commit(repo, {".env": "HARMLESS=1\n"})
    git(repo, "rm", "-q", ".env")
    git(repo, "commit", "-q", "-m", "drop .env")
    findings = cs.check_env_in_history(repo)
    assert [f.where for f in findings] == [".env"]                # one finding per path, not per commit
    assert "2 commit(s)" in findings[0].detail                    # added, then deleted
    assert cs.scan_tracked_files(repo) == []


def test_env_example_in_history_is_fine(repo):
    clean_repo(repo)
    assert cs.check_env_in_history(repo) == []


def test_history_clean_repo_has_no_findings(repo):
    assert cs.scan_history(clean_repo(repo)) == []


# ---------------------------------------------------------------- report safety and exit codes

def test_full_secret_never_appears_in_the_report(repo, capsys):
    clean_repo(repo)
    commit(repo, {"src/leak.py": f"KEY = '{FAKE_SK}'\n"})
    assert cs.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert FAKE_SK not in out and FAKE_SK[4:] not in out
    assert "sk-A... (" in out and "[FAIL] tracked" in out and "[FAIL] history" in out
    assert "RESULT: FAIL" in out


def test_no_history_flag_skips_the_history_checks(repo, capsys):
    clean_repo(repo)
    commit(repo, {"src/leak.py": f"KEY = '{FAKE_SK}'\n"})
    git(repo, "rm", "-q", "src/leak.py")
    git(repo, "commit", "-q", "-m", "remove")
    assert cs.main(["--repo", str(repo), "--no-history"]) == 0
    assert "[skip] history" in capsys.readouterr().out
    assert cs.main(["--repo", str(repo)]) == 1


def test_not_a_git_repository_returns_2(tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert cs.main(["--repo", str(plain)]) == 2
    captured = capsys.readouterr()
    assert "is not a git repository" in captured.err               # our own early check, not a later git error
    assert "[ok]" not in captured.out and "[FAIL]" not in captured.out    # no check ran