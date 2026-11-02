"""
tests/unit/test_packaging.py

Day 13 B2b: a NORMAL (non-editable) install of ragregress must contain every non-Python file the
package reads at run time, above all the report template ragregress/templates/index.html.

Why: a local editable install reads the source tree, so a missing package-data entry in
pyproject.toml stays invisible, but the Docker image installs the wheel (`pip install .`) and
`ragregress report` would fail there with TemplateNotFound.

The test copies pyproject.toml and ragregress/ to a temp dir and builds a wheel there, so the repo
gets no build/ or *.egg-info debris. It builds with the installed setuptools, or in pip's isolated build
environment if that fails; it is skipped only if neither works. Run it with -rs to see a skip reason.

Run (repo root):  python -m pytest tests/unit/test_packaging.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ragregress"
TEMPLATE = "ragregress/templates/index.html"


def non_python_files() -> set:
    return {
        p.relative_to(ROOT).as_posix()
        for p in PACKAGE.rglob("*")
        if p.is_file() and p.suffix not in {".py", ".pyc"} and "__pycache__" not in p.parts
        and ".egg-info" not in str(p)
    }


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory) -> set:
    work = tmp_path_factory.mktemp("wheel_src")
    shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
    shutil.copytree(PACKAGE, work / "ragregress", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    out = work / "dist"
    errors = []
    wheels = []
    # First with the setuptools already installed (offline); if that cannot build (e.g. no `wheel` package),
    # again in pip's isolated build environment (needs the network to fetch setuptools).
    for isolation in (["--no-build-isolation"], []):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", *isolation, "-q", "-w", str(out)],
            cwd=work, capture_output=True, text=True, timeout=600,
        )
        wheels = list(out.glob("*.whl")) if result.returncode == 0 else []
        if wheels:
            break
        errors.append(result.stderr[-200:])
    if not wheels:
        pytest.skip(f"could not build a wheel here (no setuptools/wheel and no network?): {errors}")
    return set(zipfile.ZipFile(wheels[0]).namelist())


def test_the_report_template_is_in_the_wheel(wheel_names):
    assert TEMPLATE in wheel_names, "pyproject.toml: [tool.setuptools.package-data] ragregress = [\"templates/*.html\"]"


def test_every_non_python_file_of_the_package_is_in_the_wheel(wheel_names):
    missing = sorted(non_python_files() - wheel_names)
    assert not missing, f"not packaged: {missing}"


def test_the_package_modules_are_in_the_wheel(wheel_names):
    assert "ragregress/cli.py" in wheel_names
    assert "ragregress/report.py" in wheel_names


def test_the_template_exists_in_the_source_tree():
    assert (ROOT / TEMPLATE).is_file()