"""Read-only: compare each config's Chroma collection with the chunks the current code would produce.
Usage (repo root): python scripts/check_index.py configs/baseline.yaml configs/chunk_1000.yaml
"""
import sys

from ragregress.core.config import load_run_config
from ragregress.core.runner import load_passages
from ragregress.rag.chunker import chunk_passages
from ragregress.rag.registry import build_rag_system

passages = load_passages()
for path in sys.argv[1:]:
    cfg = load_run_config(path)
    rag = build_rag_system(cfg.pipeline.id, cfg.pipeline.params)
    expected = {c.chunk_id for c in chunk_passages(passages, rag.chunker_config)}
    stored = set(rag.retriever.store.get(include=[])["ids"])
    print(f"{path}: {rag.retriever.collection_name} stored={len(stored)} expected={len(expected)} "
          f"stale={len(stored - expected)} missing={len(expected - stored)}")