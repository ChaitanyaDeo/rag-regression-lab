"""ApiSettings defaults must point at real paths in the repo.

test_api.py builds ApiSettings with temp paths, so a wrong default (Day 14: "config" instead of
"configs") passed 878 tests and only failed on a real POST /runs. Paths are relative to the
working directory; uvicorn runs from the repo root (and /work in the container).
"""
from pathlib import Path

from ragregress.api.settings import ApiSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_configs_dir_default_holds_the_shipped_configs():
    configs = REPO_ROOT / ApiSettings().configs_dir
    for name in ("baseline", "rerank_minilm"):
        assert (configs / f"{name}.yaml").is_file(), configs


def test_targets_default_is_the_shipped_targets_file():
    target = ApiSettings().targets_path
    assert target is not None
    assert (REPO_ROOT / target).is_file(), target


def test_defaults_are_relative():
    s = ApiSettings()
    for p in (s.db_path, s.configs_dir, s.passages_path, s.splits_dir, s.targets_path):
        assert not Path(p).is_absolute(), p
        