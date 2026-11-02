"""
Day 12 B4.3: configs/rerank_minilm.yaml is a one-variable change vs configs/baseline.yaml.

Guards:
  - the config loads, targets the val split (non-dev), reranker on (20 -> top_k 5)
  - the ONLY hashed difference from baseline is the added pipeline.params.reranker block
  - config_hash differs from baseline (a new run identity)
  - it uses the SAME Chroma collection as baseline (reranking never touches the index,
    F-D12-21), so the run reuses the baseline index

No model download, no Chroma client, no DB: config parsing and collection_name() only.
"""

from pathlib import Path

import pytest

from ragregress.core.config import load_run_config
from ragregress.rag.chunker import ChunkerConfig
from ragregress.rag.retriever import collection_name

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(scope="module")
def baseline():
    return load_run_config(CONFIGS / "baseline.yaml")


@pytest.fixture(scope="module")
def rerank():
    return load_run_config(CONFIGS / "rerank_minilm.yaml")


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


def test_rerank_minilm_loads_as_val_run_with_reranker_on(rerank):
    assert rerank.pipeline.id == "builtin_rag"
    assert (rerank.eval_split, rerank.is_dev) == ("val", False)
    r = rerank.pipeline.params.reranker
    assert (r.model, r.candidates) == (CROSS_ENCODER, 20)
    assert rerank.top_k == 5


def test_only_the_reranker_block_differs_from_baseline(baseline, rerank):
    a, b = _flatten(baseline.hash_payload()), _flatten(rerank.hash_payload())
    assert set(b) - set(a) == {"pipeline.params.reranker.model", "pipeline.params.reranker.candidates"}
    assert set(a) <= set(b)
    assert {k for k in a if a[k] != b[k]} == set()      # every baseline value unchanged


def test_operational_fields_match_baseline(baseline, rerank):
    assert (rerank.db_path, rerank.max_workers) == (baseline.db_path, baseline.max_workers)


def test_config_hash_differs_from_baseline(baseline, rerank):
    assert rerank.config_hash() != baseline.config_hash()


def test_uses_the_same_collection_as_baseline(baseline, rerank):
    assert _collection(rerank) == _collection(baseline)
    assert _collection(rerank).startswith("passages_all-MiniLM-L6-v2_500_")