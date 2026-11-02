"""
the optional `pipeline.params.reranker` block

Guards:
  - existing configs keep their exact config_hash (literal pins); adding the
    optional field must never move a stored run's identity 
  - "off" has exactly one serialized form: the key is absent from the hash
    payload AND from the stored run config (model_dump), whether the YAML omits
    the block or writes `reranker: null` 
  - "on" changes the hash and survives a stored-config round trip
  - validation: sub-model extra=forbid, required model, candidates bounds,
    candidates >= top_k 

No model download, no Chroma, no DB: config parsing only.
"""

import copy
from pathlib import Path

import pytest
import yaml

from ragregress.core.config import MAX_TOP_K, RunConfig, load_run_config
from ragregress.core.errors import ConfigValidationError
from ragregress.rag.registry import MAX_RERANK_CANDIDATES, RerankerParams

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
RERANKER = {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "candidates": 20}

# Full sha256 of the stored canonical runs' configs (Day 10 B12 / Day 12 B2).
PINNED_HASHES = {
    "baseline.yaml": "41c509687d97622cca1c1bdd18f4f67b801e754237ce631edbc44eab0e13f16f",
    "chunk_1000.yaml": "d0f3ccebe6967d8540e856c00dc88c9be037d561147f9b46e669f489d7a00849",
    "bge_small.yaml": "b28048b87e9016989eff6ee2030e51cb778693d5005cc62cbc181aa1687305d6",
}


def _baseline_raw():
    return yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8"))


def _raw(reranker="<absent>", top_k=None):
    raw = copy.deepcopy(_baseline_raw())
    if reranker != "<absent>":
        raw["pipeline"]["params"]["reranker"] = reranker
    if top_k is not None:
        raw["top_k"] = top_k
    return raw


def _load(tmp_path, raw):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_run_config(path)


def _load_error(tmp_path, raw):
    with pytest.raises(ConfigValidationError) as exc:
        _load(tmp_path, raw)
    return exc.value


# ---------------------------------------------------------------- hash stability (pins)

@pytest.mark.parametrize("name", sorted(PINNED_HASHES))
def test_existing_config_hash_is_pinned(name):
    assert load_run_config(CONFIGS / name).config_hash() == PINNED_HASHES[name]


# ---------------------------------------------------------------- "off" = absent everywhere

def test_reranker_defaults_to_none():
    assert load_run_config(CONFIGS / "baseline.yaml").pipeline.params.reranker is None


def test_off_is_absent_from_hash_payload_and_stored_dump():
    cfg = load_run_config(CONFIGS / "baseline.yaml")
    assert "reranker" not in cfg.hash_payload()["pipeline"]["params"]
    assert "reranker" not in cfg.model_dump(mode="json")["pipeline"]["params"]   # what Pipeline.run() stores
    assert "reranker" not in cfg.model_dump()["pipeline"]["params"]              # python mode too


def test_explicit_null_hashes_like_absent(tmp_path):
    assert _load(tmp_path, _raw(reranker=None)).config_hash() == PINNED_HASHES["baseline.yaml"]


def test_off_still_serializes_every_other_param():
    params = load_run_config(CONFIGS / "baseline.yaml").model_dump(mode="json")["pipeline"]["params"]
    assert set(params) == {"embedding_model", "chunk_size", "overlap", "generation_model", "seed"}


# ---------------------------------------------------------------- "on"

def test_on_appears_in_payload_and_changes_hash(tmp_path):
    cfg = _load(tmp_path, _raw(reranker=RERANKER))
    assert cfg.hash_payload()["pipeline"]["params"]["reranker"] == RERANKER
    assert cfg.model_dump(mode="json")["pipeline"]["params"]["reranker"] == RERANKER
    assert cfg.config_hash() != PINNED_HASHES["baseline.yaml"]


def test_reranker_settings_are_part_of_the_hash(tmp_path):
    a = _load(tmp_path, _raw(reranker=RERANKER)).config_hash()
    b = _load(tmp_path, _raw(reranker={**RERANKER, "candidates": 21})).config_hash()
    c = _load(tmp_path, _raw(reranker={**RERANKER, "model": "other/cross-encoder"})).config_hash()
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("reranker", [None, RERANKER], ids=["off", "on"])
def test_stored_config_round_trip_keeps_hash(tmp_path, reranker):
    # mirrors test_baseline_stability: stored dump -> RunConfig -> same hash
    cfg = _load(tmp_path, _raw(reranker=reranker))
    again = RunConfig.model_validate(cfg.model_dump(mode="json"))
    assert again.config_hash() == cfg.config_hash()
    assert again.pipeline.params.reranker == cfg.pipeline.params.reranker


# ---------------------------------------------------------------- validation

def test_candidates_cap_matches_max_top_k():
    assert MAX_RERANK_CANDIDATES == MAX_TOP_K


@pytest.mark.parametrize("candidates, ok", [(0, False), (1, False), (5, True), (MAX_TOP_K, True),
                                            (MAX_TOP_K + 1, False)])
def test_candidates_bounds_with_top_k_5(tmp_path, candidates, ok):
    raw = _raw(reranker={**RERANKER, "candidates": candidates})
    if ok:
        assert _load(tmp_path, raw).pipeline.params.reranker.candidates == candidates
    else:
        _load_error(tmp_path, raw)


@pytest.mark.parametrize("candidates", [0, MAX_RERANK_CANDIDATES + 1])
def test_reranker_params_bounds_on_their_own(candidates):
    # via a RunConfig, candidates >= top_k >= 1 already rejects 0; the sub-model must too
    with pytest.raises(Exception):
        RerankerParams(model=RERANKER["model"], candidates=candidates)


def test_candidates_below_top_k_is_rejected(tmp_path):
    err = _load_error(tmp_path, _raw(reranker={**RERANKER, "candidates": 4}))
    assert len(err.errors) == 1
    assert "reranker.candidates (4) must be >= top_k (5)" in err.errors[0][1]


def test_candidates_equal_top_k_is_accepted(tmp_path):
    cfg = _load(tmp_path, _raw(reranker={**RERANKER, "candidates": 10}, top_k=10))
    assert (cfg.top_k, cfg.pipeline.params.reranker.candidates) == (10, 10)


def test_unknown_reranker_key_is_rejected(tmp_path):
    err = _load_error(tmp_path, _raw(reranker={**RERANKER, "enabled": True}))
    assert [loc for loc, _ in err.errors] == ["pipeline.params.reranker.enabled"]


@pytest.mark.parametrize("missing", ["model", "candidates"])
def test_reranker_fields_are_required(tmp_path, missing):
    block = {k: v for k, v in RERANKER.items() if k != missing}
    err = _load_error(tmp_path, _raw(reranker=block))
    assert [loc for loc, _ in err.errors] == [f"pipeline.params.reranker.{missing}"]


def test_empty_model_is_rejected(tmp_path):
    _load_error(tmp_path, _raw(reranker={**RERANKER, "model": ""}))


def test_reranker_params_are_frozen():
    r = RerankerParams(**RERANKER)
    with pytest.raises(Exception):
        r.candidates = 30