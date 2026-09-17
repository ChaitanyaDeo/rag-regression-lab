"""
scripts/day4_generation_eval.py


"""

import json
from pathlib import Path
from typing import List, Dict

from ragregress.benchmark.schemas import QAPair
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.rag.generator import ABSTENTION_MESSAGE
from ragregress.rag.interface import RAGSystem
from ragregress.evaluation.metrics import token_level_prf1

from ragregress.rag.no_rag_baseline import NoRAGBaseline


DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"

def load_split(name: str) -> List[QAPair]:
    """loader.py has no split-loading helper -- read the saved split JSON
    directly, same pattern as verify_day2_ingest.py's load_passages().
    """
    raw = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]

def print_report(split_name: str, result: dict) -> None:
    print(f"\n=== {split_name} n={result['n_queries']}) ===\n")

    for row in result["rows"]:
        if row["abstained"]:
            print(f"{row['qa_id']}: ABSTAINED")
        else:
            print(
                f"{row['qa_id']}: P={row['precision']:.3f}  "
                f"R={row['recall']:.3f}  F1={row['f1']:.3f}"
            )
        print(f"    question:      {row['question']}")
        print(f"    answer_gt:     {row['answer_gt']}")
        print(f"    raw_output     {row['raw_output']}")
        print(f"    answer_pred:   {row['answer_pred']}")
        print(f"    citations:     {row['citations']}")
        print()
    
    mean_f1_display = (f"{result['mean_f1']:.3f}" if result["mean_f1"] is not None else "N/A (all abstained)")

    print(f"--- mean F1 (non-abstained, n={result['n_queries'] - result['n_abstained']}): "
        f"{mean_f1_display} ---")
    print(f"--- abstentions: {result['n_abstained']}/{result['n_queries']} "
        f"({result['abstention_rate']:.1%}) ---\n"
    )


def evaluate_generation(qa_pairs: List[QAPair], rag_system: RAGSystem, k:int)->Dict:
    """
    Runs rag_system.generate() for every QA pair, calculate score per query
    and returns both the per-query rows (for manual spot-checking) and the aggregate means.

    """

    rows: List[Dict] = []

    for qa in qa_pairs:
        # gold_ids = qa.gold_passage_ids
        # gold_set = set(gold_ids)


        contexts = rag_system.retrieve(qa.question, k = k)
        
        pred_res = rag_system.generate(query=qa.question, contexts=contexts)

        abstained = pred_res.answer.strip() == ABSTENTION_MESSAGE

        if abstained:
            precision = recall = f1_score = None
        else:
            prf1 = token_level_prf1(pred_res.answer, qa.answer)
            precision, recall, f1_score = prf1.precision, prf1.recall, prf1.f1

        rows.append(
            {
                "qa_id": qa.qa_id,
                "question": qa.question,
                "answer_gt": qa.answer,
                "raw_output": pred_res.raw_output,
                "answer_pred": pred_res.answer,
                "citations": pred_res.citations,
                "abstained": abstained,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
            }
        )

    scored_rows = [r for r in rows if not r["abstained"]]
    n_abstained = len(rows) - len(scored_rows)

    mean_f1 = (
        sum(r["f1"] for r in scored_rows) / len(scored_rows) if scored_rows else None
    )

    return {
        "n_queries": len(rows),
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / len(rows) if rows else 0.0,
        "mean_f1": mean_f1,
        "rows": rows,
    }


def print_comparison(rag_result: Dict, baseline_result: Dict) -> None:
    """Side-by-side summary of the with-RAG vs. no-RAG baseline runs.
 
    NoRAGBaseline has no abstention concept (DEC-D4-19) -- its n_abstained
    should always be 0 (barring pure coincidence), so its mean F1 is
    naturally computed over all rows, not a subset.
    """
 
    def fmt(result: Dict) -> str:
        return f"{result['mean_f1']:.3f}" if result["mean_f1"] is not None else "N/A"
 
    print("=== With-RAG vs. No-RAG baseline ===")
    print(f"  With-RAG mean F1:    {fmt(rag_result)}  "
          f"(abstentions: {rag_result['n_abstained']}/{rag_result['n_queries']})")
    print(f"  No-RAG mean F1:      {fmt(baseline_result)}  "
          f"(abstentions: {baseline_result['n_abstained']}/{baseline_result['n_queries']})")
    print()


def main() -> None:
    print("Initiate RAG system")
    system = BuiltinRAGSystem(chunker_config=ChunkerConfig(chunk_size=500, overlap=50))

    print("Loading split: dev")
    dev = load_split("dev")
    print("Loaded split")

    rag_result = evaluate_generation(dev, system, k=5)

    print_report("dev (with-RAG)", rag_result)


    baseline_system = NoRAGBaseline()
    baseline_result = evaluate_generation(dev, baseline_system, k=5)
    print_report("dev (no-RAG baseline)", baseline_result)
 
    print_comparison(rag_result, baseline_result)


if __name__ == "__main__":
    main()