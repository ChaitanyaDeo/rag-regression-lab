"""
Local execution of Pipeline


Mirrors smoke_test_evaluate.py's scale and wiring 

What this checks that inspect_graph.py couldn't:
  - the subgraph-as-node pattern (DEC-D6-12/13) actually EXECUTES, not
    just compiles
  - the run_status/query_status field split really avoids the
    concurrent-write channel collision (DEC-D6-13) -- though at
    max_workers=1 with 3 rows this mostly checks "does it run at all";
    the real concurrency proof is Task 13's 3B fan-out test
  - the retry self-loop and Send()-dispatch produce the same shape of
    result evaluate() does for the same 3 QA pairs
  - per_query_results actually reduces correctly through BOTH levels
    (subgraph's own reducer -> parent's reducer)

Run from the repo root:
 
    python scripts/smoke_test_pipeline.py

"""

import json
from pathlib import Path
from typing import List
 
from ragregress.benchmark.schemas import QAPair
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.evaluation.structured_llm import InstructorStructuredLLM
from ragregress.evaluation.metrics import RecallAtK, MRR, TokenPRF1, Faithfulness
from ragregress.pipeline.pipeline import Pipeline
 
DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
K = 5
N_SAMPLE = 3

JUDGE_MODEL = "qwen2.5:7b-instruct"

 
def load_split(name: str) -> List[QAPair]:
    raw = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]

def main() -> None:
    sample = load_split("dev")[:N_SAMPLE]
 
    builtin = BuiltinRAGSystem(ChunkerConfig(chunk_size=500, overlap=50))
    judge_llm = InstructorStructuredLLM(JUDGE_MODEL, max_retries=3)
    metrics = [
        RecallAtK(k=K),
        MRR(k=K),
        TokenPRF1(),
        Faithfulness(llm=judge_llm),
    ]
 
    pipeline = Pipeline(rag_system=builtin, metrics=metrics, k=K, max_workers=1)
 
    print(f"\n=== Pipeline smoke test: {len(sample)} rows ===\n")
    final_state = pipeline.run(qa_pairs=sample, run_id="smoke-test-001", is_dev=True)
 
    print("\n--- run_status ---")
    print(f"run_status: {final_state.get('run_status')}")
    print(f"run_status_detail: {final_state.get('run_status_detail')}")
 
    rows = final_state.get("per_query_results", [])
    print(f"\n--- per_query_results: {len(rows)} row(s) (expected {len(sample)}) ---")
    for row in rows:
        print(f"\n  [{row.get('qa_id')}] query_status={row.get('query_status')} "
              f"failure_mode={row.get('failure_mode')} last_error={row.get('last_error')}")
        for name, result in row.get("metric_results", {}).items():
            print(f"    {name}: value={result.value!r} reason={getattr(result, 'reason', None)!r}")
 
    print("\n--- aggregated_metrics ---")
    print(final_state.get("aggregated_metrics"))
 
    # Sanity checks -- fail loudly rather than requiring a human to notice
    # a subtly wrong shape.
    assert len(rows) == len(sample), (
        f"Expected {len(sample)} per_query_results rows, got {len(rows)} -- "
        f"the subgraph-as-node reducer fold-back may not be working."
    )
    qa_ids_expected = {qa.qa_id for qa in sample}
    qa_ids_got = {row.get("qa_id") for row in rows}
    assert qa_ids_expected == qa_ids_got, (
        f"qa_id mismatch: expected {qa_ids_expected}, got {qa_ids_got}"
    )
    print("\n=== Smoke test PASSED basic shape checks ===")

if __name__ == "__main__":
    main()