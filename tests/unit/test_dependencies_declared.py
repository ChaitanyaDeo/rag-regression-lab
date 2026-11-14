"""
tests/unit/test_dependencies_declared.py

Every third-party import in the repo is declared, so a fresh venv or the Docker image cannot
fail with ModuleNotFoundError at run time (e.g. `--metric rouge_l` without rouge-score).

Static: files are parsed with `ast` (nothing is imported), requirements files are read as text.

    ragregress/**                  -> requirements.txt
    tests/**, scripts/**, ./*.py   -> requirements.txt or requirements-dev.txt
    requirements.txt               -> also listed in pyproject.toml (dependencies or an extra)

Versions are not compared (pins and ranges differ on purpose). The reverse check (declared but
never imported) is not made: uvicorn, httpx and torch are used without a direct import.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, Iterable, Set

ROOT = Path(__file__).resolve().parents[2]

# import name -> distribution name, for the few that differ from "underscores to hyphens"
IMPORT_TO_DIST = {
    "yaml": "pyyaml",

    "dotenv": "python-dotenv",
    "pythonjsonlogger": "python-json-logger",
}

# Optional packages, installed by hand and deliberately not in the requirements files
# (requirements-dev.txt explains why). Keyed by file so the exception stays narrow.
OPTIONAL_IMPORTS = {
    "scripts/ragas_faithfulness_check.py": {"ragas", "langchain-community"},
}


# ---------------------------------------------------------------- helpers

def normalize(name: str) -> str:
    """PEP 503 form: lower case, runs of - _ . become one -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    """'uvicorn[standard]>=0.5 ; python_version>"3"' -> 'uvicorn'."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    if not match:
        raise ValueError(f"cannot read a package name from {spec!r}")
    return normalize(match.group(1))


def read_requirements(path: Path) -> Set[str]:
    """Package names in a requirements file; follows `-r other.txt`; skips comments and options."""
    names: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            names |= read_requirements(path.parent / line.split(None, 1)[1].strip())
        elif line.startswith("-"):
            continue                                   # --extra-index-url and similar
        else:
            names.add(requirement_name(line))
    return names


def pyproject_names(path: Path) -> Set[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    specs = list(data.get("dependencies", []))
    for extra in data.get("optional-dependencies", {}).values():
        specs.extend(extra)
    return {requirement_name(s) for s in specs}


def top_level_imports(source: str) -> Set[str]:
    """Top-level module names of every absolute import in `source`, also inside functions."""
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def first_party_names() -> Set[str]:
    names = {"ragregress", "tests", "scripts"}
    for folder in (ROOT, ROOT / "scripts", ROOT / "tests"):
        if folder.is_dir():
            names.update(p.stem for p in folder.glob("*.py"))
            names.update(p.name for p in folder.iterdir() if p.is_dir())
    return names


def third_party_dists(source: str, first_party: Set[str]) -> Set[str]:
    dists = set()
    for module in top_level_imports(source):
        if module in sys.stdlib_module_names or module in first_party:
            continue
        dists.add(IMPORT_TO_DIST.get(module, normalize(module)))
    return dists


def undeclared(files: Iterable[Path], declared: Set[str]) -> Dict[str, Set[str]]:
    """relative file path -> third-party packages it imports that are not declared."""
    first_party = first_party_names()
    problems: Dict[str, Set[str]] = {}
    for path in sorted(files):
        rel = path.relative_to(ROOT).as_posix()
        missing = third_party_dists(path.read_text(encoding="utf-8"), first_party) - declared
        missing -= OPTIONAL_IMPORTS.get(rel, set())
        if missing:
            problems[rel] = missing
    return problems


def python_files(*folders: str) -> list:
    files = []
    for name in folders:
        folder = ROOT / name
        if folder.is_dir():
            files.extend(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def describe(problems: Dict[str, Set[str]], where: str) -> str:
    lines = [f"  {rel}: {', '.join(sorted(pkgs))}" for rel, pkgs in problems.items()]
    return f"imported but not declared; add to {where}:\n" + "\n".join(lines)


# ---------------------------------------------------------------- the repo

RUNTIME = read_requirements(ROOT / "requirements.txt")
DEV = RUNTIME | read_requirements(ROOT / "requirements-dev.txt")


def test_runtime_imports_are_in_requirements_txt():
    problems = undeclared(python_files("ragregress"), RUNTIME)
    assert not problems, describe(problems, "requirements.txt and pyproject.toml")


def test_test_and_script_imports_are_declared():
    root_files = [p for p in ROOT.glob("*.py")]
    problems = undeclared(python_files("tests", "scripts") + root_files, DEV)
    assert not problems, describe(problems, "requirements-dev.txt")


def test_requirements_txt_packages_are_in_pyproject():
    missing = RUNTIME - pyproject_names(ROOT / "pyproject.toml")
    assert not missing, (
        f"in requirements.txt but not in pyproject.toml (dependencies or an extra): {sorted(missing)}"
    )


# ---------------------------------------------------------------- the checker itself

def test_the_scan_sees_the_repo():
    """A guard that scans nothing passes forever: check it finds known imports."""
    seen = set()
    first_party = first_party_names()
    for path in python_files("ragregress"):
        seen |= third_party_dists(path.read_text(encoding="utf-8"), first_party)
    assert {"pyyaml", "typer", "pydantic"} <= seen          # alias and plain names both resolved
    assert "yaml" not in seen and "os" not in seen          # import names and stdlib never leak through


def test_imports_inside_functions_and_try_blocks_are_found():
    source = (
        "import os\nfrom . import sibling\nfrom .pkg import x\n"
        "def f():\n    import rouge_score.rouge_scorer\n"
        "try:\n    from sacrebleu.metrics import BLEU\nexcept ImportError:\n    pass\n"
    )
    assert top_level_imports(source) == {"os", "rouge_score", "sacrebleu"}   # relative imports ignored


def test_requirement_lines_are_read_correctly(tmp_path):
    (tmp_path / "base.txt").write_text("PyYAML==6.0  # pinned\nuvicorn[standard]>=0.5\n", encoding="utf-8")
    (tmp_path / "dev.txt").write_text(
        "# comment\n\n-r base.txt\n--extra-index-url https://example.org/simple\n"
        "Some_Package==1.0\ntorch==2.11.0 ; python_version >= '3.11'\n", encoding="utf-8")
    assert read_requirements(tmp_path / "dev.txt") == {"pyyaml", "uvicorn", "some-package", "torch"}