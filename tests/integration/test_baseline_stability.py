"""
tests/integration/test_baseline_stability.py

Does the stored canonical baseline still reproduce with today's code? (D10-B8, plan:
test_baseline_config_metrics_stable_across_code_versions; D10-Q2, Q25-Q28)

Reads the REAL ragregress.db (read-only) and data/ files; run from the repo root.

  - Hash check (default): the stored run's config re-validates with today's RunConfig and
    re-hashes to the stored config_hash; configs/baseline.yaml hashes to the same value.
  - Re-run (opt-in, RAGREGRESS_SLOW=1, ~4 min): rebuilds the full-corpus index in a TEMP folder,
    re-runs 5 stored rows through real Ollama into a TEMP database, and requires an EXACT match
    with the stored rows (retrieved ids, answer, failure mode, every metric value).
    Rows (Q25): the lowest qa_id for each failure mode in the stored run, topped up to 5 by qa_id.
    A mismatch in retrieved ids is also the test of F-D10-26 (fresh index vs canonical index).

Skips (not fails) when ragregress.db, the reference run, or data/ files are missing.
Never writes to ragregress.db or artifacts/chroma

Run (repo root):  python -m pytest tests/integration/test_baseline_stability.py -v
       re-run:     $env:RAGREGRESS_SLOW=1; python -m pytest tests/integration/test_baseline_stability.py -v
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from ragregress.core.config import RunConfig, load_run_config
from ragregress.storage.repositories import Repository

DB_PATH = Path("ragregress.db")
REFERENCE_RUN = "val-demo-baseline-20261023-145024"    # canonical seeded baseline (D10-B12)
BASELINE_YAML = Path("configs/baseline.yaml")
N_ROWS = 5
SLOW = os.environ.get("RAGREGRESS_SLOW") == "1"


@pytest.fixture(scope="module")
def reference():
    if not DB_PATH.is_file():
        pytest.skip(f"{DB_PATH} not found (run from the repo root)")
    repo = Repository(DB_PATH)
    run = repo.get_run(REFERENCE_RUN)
    if run is None:
        pytest.skip(f"reference run {REFERENCE_RUN} not in {DB_PATH}")
    rows = {r.qa_id: r for r in repo.get_query_results(REFERENCE_RUN)}
    return run, rows


def pick_rows(rows, n=N_ROWS):
    """Lowest qa_id per failure mode present, then topped up to n by qa_id (deterministic)."""
    picked = []
    for mode in sorted({r.failure_mode for r in rows.values() if r.failure_mode}):
        picked.append(min(q for q, r in rows.items() if r.failure_mode == mode))
    for q in sorted(rows):
        if len(picked) >= n:
            break
        if q not in picked:
            picked.append(q)
    return sorted(picked[:n])


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def values(row):
    return {name: res.get("value") for name, res in (row.metric_results or {}).items()}


# ---------------------------------------------------------------- hash check (default)

def test_stored_config_rehashes_to_stored_hash(reference):
    run, _ = reference
    cfg = RunConfig.model_validate(run.config)
    assert cfg.config_hash() == run.config_hash
    assert (cfg.eval_split, cfg.is_dev) == ("val", False)


def test_baseline_yaml_is_the_reference_config(reference):
    run, _ = reference
    assert load_run_config(BASELINE_YAML).config_hash() == run.config_hash


def test_reference_run_is_complete(reference):
    run, rows = reference
    assert run.run_status == "healthy"
    assert len(rows) == 50
    assert len(pick_rows(rows)) == N_ROWS


# ---------------------------------------------------------------- re-run (opt-in)

@pytest.mark.skipif(not SLOW, reason="real Ollama re-run; set RAGREGRESS_SLOW=1")
def test_baseline_config_metrics_stable_across_code_versions(reference, tmp_path):
    from ragregress.core.errors import ConfigValidationError
    from ragregress.core.runner import build_metrics, load_passages, load_qa_pairs
    from ragregress.pipeline.pipeline import Pipeline
    from ragregress.rag.builtin_system import BuiltinRAGSystem
    from ragregress.rag.chunker import ChunkerConfig
    from ragregress.rag.generator import GeneratorConfig

    run, stored = reference
    cfg = RunConfig.model_validate(run.config)
    try:
        passages = load_passages()
        qa_all = {q.qa_id: q for q in load_qa_pairs(cfg.split_path())}
    except ConfigValidationError as exc:
        pytest.skip(str(exc))

    chosen = pick_rows(stored)
    p = cfg.pipeline.params
    # Mirrors registry._build_builtin_rag, but with a temp persist_directory (never the real index).
    system = BuiltinRAGSystem(
        ChunkerConfig(chunk_size=p.chunk_size, overlap=p.overlap),
        generator_config=GeneratorConfig(model_name=p.generation_model, seed=p.seed),
        embedding_model=p.embedding_model,
        persist_directory=str(tmp_path / "chroma"),
    )
    system.ensure_indexed(passages)

    db = tmp_path / "stability.db"                      # never the real ragregress.db
    pipe = Pipeline(rag_system=system, metrics=build_metrics(cfg), k=cfg.top_k, max_workers=1, db_path=db)
    pipe.run(qa_pairs=[qa_all[q] for q in chosen], run_id="stability", is_dev=False)
    fresh = {r.qa_id: r for r in Repository(db).get_query_results("stability")}

    mismatches = []
    for q in chosen:
        old, new = stored[q], fresh.get(q)
        if new is None:
            mismatches.append(f"{q}: missing from re-run")
            continue
        for label, a, b in (
            ("retrieved_ids", old.retrieved_ids, new.retrieved_ids),
            ("answer", old.generation_answer, new.generation_answer),
            ("failure_mode", old.failure_mode, new.failure_mode),
            ("query_status", old.query_status, new.query_status),
        ):
            if a != b:
                mismatches.append(f"{q} {label}: stored={a!r} now={b!r}")
        ov, nv = values(old), values(new)
        for name in sorted(set(ov) | set(nv)):
            if not _same(ov.get(name), nv.get(name)):
                mismatches.append(f"{q} {name}: stored={ov.get(name)!r} now={nv.get(name)!r}")

    assert not mismatches, "baseline drifted:\n  " + "\n  ".join(mismatches)