"""
scripts/inspect_context_recall.py

Re-runs context_recall on a few rows of a config's split and prints what the judge saw and said:
the retrieved chunks, the reference built from the gold answers, and every statement with its verdict and reason.
Use it to judge the metric's quality; a run stores only the score, not these details.

    python scripts/inspect_context_recall.py --config configs/dev.yaml --n 5
    python scripts/inspect_context_recall.py --config configs/dev.yaml --match "SAP Center"

Needs Ollama (or the configured judge) running. Run from the repository root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from ragregress.core.config import load_run_config
from ragregress.core.runner import PASSAGES_PATH, load_passages, load_qa_pairs
from ragregress.evaluation.judge import build_judge
from ragregress.evaluation.metrics.context_recall.metric import ContextRecall
from ragregress.evaluation.reference import build_reference
from ragregress.rag.registry import build_rag_system


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5, help="rows to inspect (default 5)")
    parser.add_argument("--match", default=None, help="only rows whose question contains this text")
    parser.add_argument("--chars", type=int, default=300, help="characters of each chunk to print")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    cfg = load_run_config(args.config)
    qa_pairs = load_qa_pairs(cfg.split_path())
    if args.match:
        qa_pairs = [qa for qa in qa_pairs if args.match.lower() in qa.question.lower()]
    qa_pairs = qa_pairs[: args.n]

    rag = build_rag_system(cfg.pipeline.id, cfg.pipeline.params)
    if hasattr(rag, "ensure_indexed"):
        rag.ensure_indexed(load_passages(PASSAGES_PATH))
    judge = build_judge(cfg.judge_provider, cfg.judge_model, seed=cfg.judge_seed, max_retries=3)
    metric = ContextRecall(llm=judge)

    for qa in qa_pairs:
        chunks = rag.retrieve(qa.question, k=cfg.top_k)
        reference = build_reference(qa.answer)
        result = metric.score(question=qa.question, reference=reference, context=chunks)

        print("=" * 100)
        print("QUESTION  :", qa.question)
        print("GOLD IDS  :", list(qa.gold_passage_ids), "| RETRIEVED:", [c.source_passage_id for c in chunks])
        print("REFERENCE :", reference.replace("\n", " | ") or "(empty)")
        for i, chunk in enumerate(chunks, 1):
            print(f"  chunk {i} [{chunk.source_passage_id}]: {chunk.text[: args.chars]!r}")
        print("SCORE     :", result.value, result.reason or "")
        for s in result.details.get("statements", []):
            print(f"  {'SUPPORTED  ' if s['verdict'] else 'UNSUPPORTED'} {s['statement']!r}\n      reason: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())