"""
scripts/day3_retrieval_eval.py

Day 3 deliverable: computes Recall@5 and MRR for the baseline
BuiltinRAGSystem config (chunk_size=500, overlap=50, MiniLM) against the
dev (15 questions) and val (50 questions) splits.

Does NOT re-ingest -- reuses the Chroma collection already built and
verified in Day 2 (scripts/verify_day2_ingest.py), since retrieve() reads
from whatever collection_name(embedding_model, chunker_config) resolves
to, and that collection already exists on disk from Day 2.

Prints per-query rows (question, gold_passage_id, retrieved ids, rank) so
at least 3-5 can be checked by hand against the printed Recall@5/MRR,
per the Day 3 definition of done -- then the aggregate numbers for dev
and val.

"""

import json
from pathlib import Path
from typing import List

from ragregress.benchmark.schemas import QAPair
from ragregress.evaluation.retrieval_eval import evaluate_retrieval
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig


DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
K = 5

def load_split(name: str) -> List[QAPair]:
    """loader.py has no split-loading helper -- read the saved split JSON
    directly, same pattern as verify_day2_ingest.py's load_passages().
    """
    raw = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]

def print_report(split_name: str, result: dict) -> None:
    print(f"\n=== {split_name} (k={result['k']}, n={result['n_queries']}) ===\n")
 
    for row in result["rows"]:
        rank_display = row["rank"] if row["rank"] is not None else "not in top-k"
        print(f"  [{row['qa_id']}] rank={rank_display}  recall@{K}={row['recall_at_k']:.3f}  mrr={row['mrr']:.3f}")
        print(f"    question:          {row['question']}")
        print(f"    gold_passage_ids:  {row['gold_passage_ids']}")
        print(f"    retrieved_ids:     {row['retrieved_ids']}")
        print()
 
    print(f"  Recall@{result['k']}: {result['mean_recall_at_k']:.3f}")
    print(f"  MRR:       {result['mean_mrr']:.3f}")


def main() -> None:
    system = BuiltinRAGSystem(chunker_config=ChunkerConfig(chunk_size=500, overlap=50))

    dev = load_split("dev")
    val = load_split("val")

    dev_result = evaluate_retrieval(dev, system, k=K)
    print_report("dev", dev_result)

    val_result = evaluate_retrieval(val, system, k=K)
    print_report("val", val_result)


if __name__ == "__main__":
    main()
