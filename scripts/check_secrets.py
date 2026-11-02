"""
scripts/check_secrets.py

secrets-exposure check for this repository. Standard library only, needs `git`.

    python scripts/check_secrets.py                  # all checks, whole history
    python scripts/check_secrets.py --no-history     # skip the (slowest) history scan
    python scripts/check_secrets.py --repo <path>

Checks
  gitignore    .env and its variants (.env.local, .env.production, .env.development) are ignored;
               .env.example is NOT ignored
  env-example  .env.example holds placeholders only: no secret shapes, no real-looking values
  tracked      no tracked file contains a secret shape; no .env file (except .env.example) is tracked
  env-history  no .env file (except .env.example) was ever committed
  history      no commit ever ADDED a line containing a secret shape (`git log -p --all`)

"Secret shape" = high-precision patterns only (OpenAI-style sk-, GitHub ghp_..., HuggingFace hf_,
AWS AKIA..., Slack xox..., private key blocks). Findings never print the full match
(first 4 characters + length), so the report itself is safe to paste anywhere.

Exit code: 0 clean, 1 findings, 2 cannot run (git missing / not a repository).

Known limits: a secret with no recognisable shape and not in a .env file is not found; a secret that
already left the machine (pushed) must be rotated, not just removed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

SECRET_PATTERNS = [
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("huggingface token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
]

MUST_BE_IGNORED = (".env", ".env.local", ".env.production", ".env.development")
MUST_NOT_BE_IGNORED = (".env.example",)
EXAMPLE_NAME = ".env.example"

MAX_FILE_BYTES = 20 * 1024 * 1024

_PLACEHOLDER_WORDS = {
    "", "ollama", "changeme", "change-me", "change_me", "none", "null", "true", "false",
    "debug", "info", "warning", "error", "critical",
}


@dataclass(frozen=True)
class Finding:
    check: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.where}: {self.detail}"


class GitError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers

def mask(secret: str) -> str:
    return f"{secret[:4]}... ({len(secret)} chars)"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def ensure_repo(repo: Path) -> None:
    proc = _git(repo, "rev-parse", "--is-inside-work-tree", check=False)
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise GitError(f"{repo} is not a git repository")


def secret_matches(line: str):
    """Yield (pattern_name, matched_text) for every secret shape in `line`."""
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(line):
            yield name, match.group(0)


def is_env_secret_file(path: str) -> bool:
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    return base.startswith(".env") and base != EXAMPLE_NAME


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip("\"'")
    if v.lower() in _PLACEHOLDER_WORDS:
        return True
    if re.fullmatch(r"<[^>]+>", v) or re.fullmatch(r"\$\{[^}]+\}", v):
        return True
    if re.match(r"(?i)(your|my)[-_ ]", v) or re.fullmatch(r"(?i)x{3,}|\*{3,}", v):
        return True
    if re.fullmatch(r"\d+", v):
        return True
    if re.fullmatch(r"https?://[^\s@/]+(:\d+)?(/\S*)?", v):          # a URL without user:password@
        return True
    return False


# ---------------------------------------------------------------- checks

def check_gitignore(repo: Path) -> List[Finding]:
    findings = []

    def ignored(path: str) -> bool:
        return _git(repo, "check-ignore", "--no-index", "-q", "--", path, check=False).returncode == 0

    for path in MUST_BE_IGNORED:
        if not ignored(path):
            findings.append(Finding("gitignore", path, "is not ignored by .gitignore"))
    for path in MUST_NOT_BE_IGNORED:
        if ignored(path):
            findings.append(Finding("gitignore", path, "is ignored, but must stay trackable"))
    return findings


def check_env_example(repo: Path) -> List[Finding]:
    path = repo / EXAMPLE_NAME
    if not path.is_file():
        return [Finding("env-example", EXAMPLE_NAME, "file is missing")]

    findings = []
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for name, text in secret_matches(line):
            findings.append(Finding("env-example", f"{EXAMPLE_NAME}:{number}", f"{name} {mask(text)}"))
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if sep and not _is_placeholder(value):
            findings.append(Finding("env-example", f"{EXAMPLE_NAME}:{number}",
                                    f"value of {key.strip()} does not look like a placeholder {mask(value.strip())}"))
    return findings


def _tracked_files(repo: Path) -> List[str]:
    out = _git(repo, "ls-files", "-z").stdout
    return [p for p in out.split("\0") if p]


def scan_tracked_files(repo: Path) -> List[Finding]:
    findings = []
    for rel in _tracked_files(repo):
        if is_env_secret_file(rel):
            findings.append(Finding("tracked", rel, "an .env file is tracked by git"))
        path = repo / rel
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue                                         # binary
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            for name, text in secret_matches(line):
                findings.append(Finding("tracked", f"{rel}:{number}", f"{name} {mask(text)}"))
    return findings


def check_env_in_history(repo: Path) -> List[Finding]:
    proc = _git(repo, "log", "--all", "--full-history", "--name-only", "--format=commit %H")
    commits_by_path: dict = {}
    commit = None
    for line in proc.stdout.splitlines():                    # newest commit first
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.strip() and is_env_secret_file(line.strip()):
            commits_by_path.setdefault(line.strip(), []).append(commit)
    return [
        Finding("env-history", path,
                f"an .env file was committed at some point ({len(set(commits))} commit(s), newest {commits[0][:10]})")
        for path, commits in commits_by_path.items()
    ]


def scan_history(repo: Path) -> List[Finding]:
    """Every line ever ADDED by any commit on any branch, scanned for secret shapes."""
    proc = _git(repo, "log", "-p", "--all", "--no-color", "--format=commit %H")
    findings, seen = [], set()
    commit, path = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("commit ") and len(line.split()) == 2 and len(line.split()[1]) == 40:
            commit = line.split()[1]
        elif line.startswith("+++ "):
            target = line[4:].strip()
            path = target[2:] if target.startswith("b/") else path
        elif line.startswith("--- "):
            continue
        elif line.startswith("+"):
            for name, text in secret_matches(line[1:]):
                key = (commit, path, name)
                if key not in seen:
                    seen.add(key)
                    findings.append(Finding("history", f"{(commit or '?')[:10]}:{path}", f"{name} {mask(text)}"))
    return findings


CHECKS = (
    ("gitignore", check_gitignore),
    ("env-example", check_env_example),
    ("tracked", scan_tracked_files),
    ("env-history", check_env_in_history),
    ("history", scan_history),
)


def run_all(repo: Path, history: bool = True) -> List[Finding]:
    ensure_repo(repo)
    findings: List[Finding] = []
    for name, check in CHECKS:
        if name in ("env-history", "history") and not history:
            continue
        findings.extend(check(repo))
    return findings


# ---------------------------------------------------------------- CLI

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Secrets-exposure check for a git repository.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root (default: current directory)")
    parser.add_argument("--no-history", action="store_true", help="skip the env-history and history scans")
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo = args.repo.resolve()

    print(f"secrets check: {repo}")
    try:
        ensure_repo(repo)
        total = 0
        for name, check in CHECKS:
            if name in ("env-history", "history") and args.no_history:
                print(f"[skip] {name}")
                continue
            findings = check(repo)
            total += len(findings)
            print(f"[{'ok' if not findings else 'FAIL'}] {name}" + (f" ({len(findings)})" if findings else ""))
            for finding in findings:
                print(f"       {finding}")
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("RESULT: " + ("PASS" if total == 0 else f"FAIL ({total} finding(s))"))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())