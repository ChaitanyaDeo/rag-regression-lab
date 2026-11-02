"""
Day 12 B1: configs/bge_small.yaml is a one-variable change vs configs/baseline.yaml.

Guards:
  - the config loads and targets the val split (non-dev)
  - the ONLY hashed difference from baseline is pipeline.params.embedding_model
  - config_hash differs from baseline (a new, distinct run identity)
  - its Chroma collection name differs from baseline's (no index collision,
    F-D10-33) while the chunker fingerprint is identical, and both names are
    legal Chroma collection names

No model download, no Chroma client, no DB: only config parsing and the
pure collection_name() function.
"""

import re
from pathlib import Path

import pytest

from ragregress.core.config import load_run_config
from ragregress.rag.chunker import ChunkerConfig
from ragregress.rag.retriever import collection_name

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
BGE_ID = "BAAI/bge-small-en-v1.5"

# Chroma rule: 3-63 chars, [a-zA-Z0-9._-], starts and ends alphanumeric
CHROMA_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


@pytest.fixture(scope="module")
def baseline():
    return load_run_config(CONFIGS / "baseline.yaml")


@pytest.fixture(scope="module")
def bge():
    return load_run_config(CONFIGS / "bge_small.yaml")


def _flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


def _collection(cfg):
    p = cfg.pipeline.params
    return collection_name(p.embedding_model, ChunkerConfig(chunk_size=p.chunk_size, overlap=p.overlap))


def test_bge_small_loads_as_val_run(bge):
    assert bge.pipeline.id == "builtin_rag"
    assert bge.pipeline.params.embedding_model == BGE_ID
    assert (bge.eval_split, bge.is_dev) == ("val", False)


def test_only_embedding_model_differs_from_baseline(baseline, bge):
    a, b = _flatten(baseline.hash_payload()), _flatten(bge.hash_payload())
    assert a.keys() == b.keys()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"pipeline.params.embedding_model"}


def test_operational_fields_match_baseline(baseline, bge):
    # not hashed, but a different db_path would put the run in another DB
    assert (bge.db_path, bge.max_workers) == (baseline.db_path, baseline.max_workers)


def test_config_hash_differs_from_baseline(baseline, bge):
    assert bge.config_hash() != baseline.config_hash()


def test_collection_names_distinct_same_chunker_fingerprint(baseline, bge):
    base_name, bge_name = _collection(baseline), _collection(bge)
    assert bge_name != base_name
    assert bge_name.startswith("passages_bge-small-en-v1.5_500_")
    # last segment = chunker fingerprint: identical chunking, only the model differs
    assert bge_name.rsplit("_", 1)[1] == base_name.rsplit("_", 1)[1]


@pytest.mark.parametrize("name", ["baseline.yaml", "bge_small.yaml"])
def test_collection_name_is_chroma_legal(name):
    cfg = load_run_config(CONFIGS / name)
    assert CHROMA_NAME.fullmatch(_collection(cfg))