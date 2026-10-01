"""
scripts/ragas_faithfulness_check.py

This is a standalone sanity-check script, not part of the ragregress
package or the pipeline -- per the plan (RAG_Regression_Lab_Plan.md):
"Ragas [is] used only as a second-opinion sanity check on ~10 examples to
validate implementation.
"""
from __future__ import annotations
 
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
 
from openai import AsyncOpenAI
 
from ragregress.benchmark.schemas import QAPair
from ragregress.evaluation.metrics import Faithfulness
from ragregress.evaluation.structured_llm import InstructorStructuredLLM
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
 
DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
OUT_DIR = Path("results")
 
DEFAULT_N = 10
DEFAULT_K = 5
DEFAULT_JUDGE_MODEL = "qwen2.5:7b-instruct"
OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"
 
# Rows whose |our_score - ragas_score| exceeds this are flagged as a
# "large discrepancy" for the README's limitations section (DoD wording).
DISCREPANCY_THRESHOLD = 0.3
 
 
def load_split(name: str) -> List[QAPair]:
    raw = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=str, default="dev", help="Split name under data/splits/ (default: dev).")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Number of successful comparisons to collect (default: {DEFAULT_N}).")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Top-k for retrieval (default: {DEFAULT_K}).")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help=f"Ollama model name, used as the judge for BOTH metrics (default: {DEFAULT_JUDGE_MODEL}).")
    return parser.parse_args()
 
 
async def _score_ragas(ragas_metric, question: str, answer: str, contexts: List[str]):
    return await ragas_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
 
 
async def run_comparisons(qa_pairs: List[QAPair], n: int, k: int, judge_model: str) -> List[Dict[str, Any]]:
    # Same BuiltinRAGSystem as everywhere else in this project (run_split.py,
    # dev_faithfulness_eval.py) -- chunk_size=500, overlap=50.
    rag_system = BuiltinRAGSystem(ChunkerConfig(chunk_size=500, overlap=50))
 
    # Our metric, via the same InstructorStructuredLLM wrapper the real
    # pipeline uses.
    our_llm = InstructorStructuredLLM(judge_model, max_retries=3)
    our_faithfulness = Faithfulness(llm=our_llm)
 
    # Ragas's metric, pointed at the SAME model via Ollama's OpenAI-compat
    # endpoint (see module docstring for why this differs from
    # InstructorStructuredLLM's "ollama/<model>" provider shorthand).
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness as RagasFaithfulness
 
    ollama_client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_OPENAI_BASE_URL)
    ragas_llm = llm_factory(judge_model, provider="openai", client=ollama_client)
    ragas_faithfulness = RagasFaithfulness(llm=ragas_llm)
 
    results: List[Dict[str, Any]] = []
    skipped = 0
 
    for qa in qa_pairs:
        if len(results) >= n:
            break
 
        chunks = rag_system.retrieve(qa.question, k=k)
        if not chunks:
            skipped += 1
            print(f"  [skip] {qa.qa_id}: no chunks retrieved, nothing to score")
            continue
 
        generation = rag_system.generate(qa.question, chunks)
        context_texts = [c.text for c in chunks]
 
        try:
            our_result = our_faithfulness.score(question=qa.question, answer=generation.answer, context=chunks)
        except Exception as e:
            skipped += 1
            print(f"  [skip] {qa.qa_id}: our Faithfulness.score() raised {e!r}")
            continue
 
        try:
            ragas_result = await _score_ragas(ragas_faithfulness, qa.question, generation.answer, context_texts)
        except Exception as e:
            skipped += 1
            print(f"  [skip] {qa.qa_id}: ragas ascore() raised {e!r}")
            continue
 
        our_score = our_result.value
        ragas_score = ragas_result.value
        diff = abs(our_score - ragas_score) if (our_score == our_score and ragas_score == ragas_score) else float("nan")
 
        row = {
            "qa_id": qa.qa_id,
            "question": qa.question,
            "answer": generation.answer,
            "our_score": our_score,
            "our_reason": our_result.reason,
            "ragas_score": ragas_score,
            "ragas_reason": getattr(ragas_result, "reason", None),
            "diff": diff,
            "large_discrepancy": bool(diff == diff and diff > DISCREPANCY_THRESHOLD),
        }
        results.append(row)
 
        print(
            f"  [{len(results)}/{n}] {qa.qa_id} -- our={our_score:.3f} ragas={ragas_score:.3f} "
            f"diff={diff:.3f}{' *** LARGE DISCREPANCY ***' if row['large_discrepancy'] else ''}"
        )
 
    if len(results) < n:
        print(f"\nWarning: only found {len(results)}/{n} scoreable rows (skipped {skipped}) before the split ran out.")
 
    return results
 
 
def print_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("\nNo comparisons collected -- nothing to summarize.")
        return
 
    our_scores = [r["our_score"] for r in results if r["our_score"] == r["our_score"]]
    ragas_scores = [r["ragas_score"] for r in results if r["ragas_score"] == r["ragas_score"]]
    diffs = [r["diff"] for r in results if r["diff"] == r["diff"]]
    large = [r for r in results if r["large_discrepancy"]]
 
    print(f"\n=== summary ({len(results)} rows) ===")
    print(f"  mean our_score:   {sum(our_scores) / len(our_scores):.4f}" if our_scores else "  mean our_score:   n/a (all NaN)")
    print(f"  mean ragas_score: {sum(ragas_scores) / len(ragas_scores):.4f}" if ragas_scores else "  mean ragas_score: n/a (all NaN)")
    print(f"  mean |diff|:      {sum(diffs) / len(diffs):.4f}" if diffs else "  mean |diff|:      n/a")
    print(f"  large discrepancies (|diff| > {DISCREPANCY_THRESHOLD}): {len(large)}/{len(results)}")
    for r in large:
        print(f"    - {r['qa_id']}: our={r['our_score']:.3f} ragas={r['ragas_score']:.3f}")
 
 
def main() -> None:
    args = parse_args()
 
    qa_pairs = load_split(args.split)
    print(f"Loaded split '{args.split}': {len(qa_pairs)} QA pairs available, collecting {args.n} comparisons "
          f"(judge model: {args.judge_model})\n")
 
    results = asyncio.run(run_comparisons(qa_pairs, n=args.n, k=args.k, judge_model=args.judge_model))
 
    print_summary(results)
 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ragas_faithfulness_check.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote full comparison detail -> {out_path}")
 
 
if __name__ == "__main__":
    main()