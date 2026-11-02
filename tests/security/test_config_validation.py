"""
tests/security/test_config_validation.py

Hostile / malformed run configs are rejected at the boundary.

Two layers:
  - unit: load_run_config(tmp yaml) -> ConfigValidationError with the exact
    (location, message) pair and exit_code 3
  - CLI:  `run --config <tmp> --dry-run` -> exit 3, one readable error, no traceback

Safety (F-D10-33): every config lives in tmp_path and sets db_path inside it;
the CLI is always invoked with --dry-run, so even a config that wrongly passes
validation executes nothing and touches no real DB / Chroma folder.

Pinned on purpose (D11-Q7 = A): pydantic lax int coercion is current
behaviour (`true` -> 1, "5" -> 5, 1.0e+3 -> 1000); see test_lax_int_coercion_*.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragregress.cli import app
from ragregress.core.config import MAX_TOP_K, MAX_WORKERS_CAP, load_run_config
from ragregress.core.errors import ConfigValidationError
from ragregress.rag.registry import MAX_CHUNK_SIZE

# Mirrors configs/baseline.yaml (db_path is replaced per test with a tmp path).
BASE = {
    "pipeline": {
        "id": "builtin_rag",
        "params": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 500,
            "overlap": 50,
            "generation_model": "qwen2.5:3b-instruct",
            "seed": 42,
        },
    },
    "top_k": 5,
    "judge_model": "qwen2.5:7b-instruct",
    "judge_seed": 42,
    "eval_split": "val",
    "is_dev": False,
    "max_workers": 1,
}


# ---------------------------------------------------------------- helpers

def make_cfg(**changes):
    """Copy of BASE with dotted-path changes; a value of DELETE removes the key."""
    cfg = copy.deepcopy(BASE)
    for dotted, value in changes.items():
        *parents, leaf = dotted.split("__")
        node = cfg
        for part in parents:
            node = node[part]
        if value is DELETE:
            del node[leaf]
        else:
            node[leaf] = value
    return cfg


DELETE = object()


def write_yaml(tmp_path: Path, cfg) -> Path:
    if isinstance(cfg, dict):
        cfg = {**cfg, "db_path": str(tmp_path / "test.db")}
        text = yaml.safe_dump(cfg, sort_keys=False)
    else:
        text = cfg                       # raw text for malformed-YAML cases
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load_error(tmp_path, cfg) -> ConfigValidationError:
    with pytest.raises(ConfigValidationError) as exc_info:
        load_run_config(write_yaml(tmp_path, cfg))
    assert exc_info.value.exit_code == 3
    return exc_info.value


def locs(err: ConfigValidationError):
    return [loc for loc, _ in err.errors]


def run_cli(path: Path):
    return CliRunner().invoke(app, ["run", "--config", str(path), "--dry-run"])


# ---------------------------------------------------------------- sanity

def test_base_config_is_valid(tmp_path):
    cfg = load_run_config(write_yaml(tmp_path, make_cfg()))
    assert cfg.top_k == 5 and cfg.pipeline.params.chunk_size == 500


def test_cli_dry_run_accepts_base_config_and_runs_nothing(tmp_path):
    result = run_cli(write_yaml(tmp_path, make_cfg()))
    assert result.exit_code == 0, result.output
    assert "dry run: config is valid, nothing was executed" in result.output
    assert not (tmp_path / "test.db").exists()


# ---------------------------------------------------------------- plan items

def test_missing_chunk_size_is_reported_by_field(tmp_path):
    err = load_error(tmp_path, make_cfg(pipeline__params__chunk_size=DELETE))
    assert ("pipeline.params.chunk_size", "Field required") in err.errors
    assert "1 validation error(s)" in str(err)


def test_huge_top_k_is_rejected(tmp_path):
    err = load_error(tmp_path, make_cfg(top_k=100000))
    assert locs(err) == ["top_k"]
    assert f"less than or equal to {MAX_TOP_K}" in err.errors[0][1]


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"pipeline__params__chunk_size": DELETE}, "pipeline.params.chunk_size"),
        ({"top_k": 100000}, "top_k"),
    ],
)
def test_cli_rejects_with_exit_3_and_no_traceback(tmp_path, changes, field):
    path = write_yaml(tmp_path, make_cfg(**changes))
    result = run_cli(path)
    assert result.exit_code == 3, result.output
    assert result.output.startswith("error: ")
    assert f"  - {field}: " in result.output
    assert "Traceback" not in result.output
    assert "dry run" not in result.output
    assert not (tmp_path / "test.db").exists()


# ---------------------------------------------------------------- caps and boundaries

@pytest.mark.parametrize("value, ok", [(0, False), (1, True), (MAX_TOP_K, True), (MAX_TOP_K + 1, False)])
def test_top_k_bounds(tmp_path, value, ok):
    cfg = make_cfg(top_k=value)
    if ok:
        assert load_run_config(write_yaml(tmp_path, cfg)).top_k == value
    else:
        assert locs(load_error(tmp_path, cfg)) == ["top_k"]


@pytest.mark.parametrize("value, ok", [(MAX_CHUNK_SIZE, True), (MAX_CHUNK_SIZE + 1, False)])
def test_chunk_size_cap(tmp_path, value, ok):
    cfg = make_cfg(pipeline__params__chunk_size=value)
    if ok:
        assert load_run_config(write_yaml(tmp_path, cfg)).pipeline.params.chunk_size == value
    else:
        err = load_error(tmp_path, cfg)
        assert locs(err) == ["pipeline.params.chunk_size"]
        assert f"less than or equal to {MAX_CHUNK_SIZE}" in err.errors[0][1]


@pytest.mark.parametrize("overlap", [500, 501])
def test_overlap_not_less_than_chunk_size_is_rejected(tmp_path, overlap):
    err = load_error(tmp_path, make_cfg(pipeline__params__overlap=overlap))
    assert locs(err) == ["pipeline.params"]
    assert f"overlap ({overlap}) must be less than chunk_size (500)" in err.errors[0][1]


@pytest.mark.parametrize("value", [0, MAX_WORKERS_CAP + 1])
def test_max_workers_bounds(tmp_path, value):
    assert locs(load_error(tmp_path, make_cfg(max_workers=value))) == ["max_workers"]


@pytest.mark.parametrize("dotted, loc", [("pipeline__params__seed", "pipeline.params.seed"),
                                         ("judge_seed", "judge_seed")])
def test_negative_seed_is_rejected(tmp_path, dotted, loc):
    assert locs(load_error(tmp_path, make_cfg(**{dotted: -1}))) == [loc]


# ---------------------------------------------------------------- unknown / inconsistent fields

@pytest.mark.parametrize(
    "dotted, loc",
    [
        ("pipeline__params__chunck_size", "pipeline.params.chunck_size"),   # typo
        ("extra", "extra"),                                                  # top level
        ("pipeline__extra", "pipeline.extra"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path, dotted, loc):
    err = load_error(tmp_path, make_cfg(**{dotted: 1}))
    assert (loc, "Extra inputs are not permitted") in err.errors


def test_unknown_pipeline_id_is_rejected(tmp_path):
    err = load_error(tmp_path, make_cfg(pipeline__id="evil_rag"))
    assert locs(err) == ["pipeline.params"]
    assert "unknown pipeline id 'evil_rag'" in err.errors[0][1]


def test_is_dev_must_match_eval_split(tmp_path):
    err = load_error(tmp_path, make_cfg(is_dev=True))
    assert "is_dev=True does not match eval_split='val'" in str(err)


def test_all_errors_reported_at_once(tmp_path):
    err = load_error(tmp_path, make_cfg(top_k=100000, max_workers=99))
    assert sorted(locs(err)) == ["max_workers", "top_k"]
    assert "2 validation error(s)" in str(err)


# ---------------------------------------------------------------- file / YAML level

@pytest.mark.parametrize(
    "text, fragment",
    [
        ("top_k: [5\n", "invalid YAML"),
        ("- just\n- a list\n", "top level of the config must be a mapping"),
        ("", "top level of the config must be a mapping"),
    ],
)
def test_malformed_yaml_is_rejected(tmp_path, text, fragment):
    err = load_error(tmp_path, text)
    assert fragment in str(err)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigValidationError, match="config file not found"):
        load_run_config(tmp_path / "nope.yaml")
    result = run_cli(tmp_path / "nope.yaml")
    assert result.exit_code == 3 and "config file not found" in result.output


def test_yaml_python_tag_is_not_executed(tmp_path):
    """safe_load must refuse python/object tags: the payload would create `pwned`."""
    marker = tmp_path / "pwned"
    text = f"top_k: !!python/object/apply:builtins.open ['{marker}', 'w']\n"
    path = write_yaml(tmp_path, text)

    err = load_error(tmp_path, text)
    assert "invalid YAML" in str(err)
    result = run_cli(path)
    assert result.exit_code == 3 and "invalid YAML" in result.output
    assert not marker.exists()


# ---------------------------------------------------------------- pinned lax coercion (D11-Q7 = A)

@pytest.mark.parametrize(
    "dotted, value, expected",
    [("top_k", True, 1), ("top_k", "5", 5), ("pipeline__params__chunk_size", 1000.0, 1000)],
)
def test_lax_int_coercion_is_current_behaviour(tmp_path, dotted, value, expected):
    dumped = load_run_config(write_yaml(tmp_path, make_cfg(**{dotted: value}))).model_dump()
    node = dumped
    for part in dotted.split("__"):
        node = node[part]
    assert node == expected and type(node) is int


@pytest.mark.parametrize("raw", ["5.5", "1e3"])   # 1e3 is a YAML string under safe_load
def test_non_integral_top_k_is_rejected(tmp_path, raw):
    text = yaml.safe_dump({**BASE, "db_path": str(tmp_path / "t.db")}, sort_keys=False)
    text = text.replace("top_k: 5", f"top_k: {raw}")
    assert locs(load_error(tmp_path, text)) == ["top_k"]