"""
Runs the real LangGraph Pipeline 

Usage:
    python scripts/run_split.py --split dev
    python scripts/run_split.py --split val
    python scripts/run_split.py --split-path data/splits/custom.json
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
 
from ragregress.benchmark.schemas import QAPair
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.evaluation.structured_llm import InstructorStructuredLLM
from ragregress.evaluation.metrics import RecallAtK, MRR, TokenPRF1, Faithfulness
from ragregress.pipeline.pipeline import Pipeline
from ragregress.storage.database import DB_PATH_DEFAULT

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_K = 5
DEFAULT_MAX_WORKERS = 1
DEFAULT_JUDGE_MODEL = "qwen2.5:7b-instruct"

def load_split(path: Path) -> List[QAPair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]

def resolve_split(split_name: Optional[str], split_path: Optional[str]) -> tuple[Path, str]:
    """Returns (path_to_load, a short name used for run_id/is_dev inference)."""
    if split_path:
        path = Path(split_path)
        return path, path.stem
    name = split_name or "dev"
    return SPLITS_DIR / f"{name}.json", name

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split", type=str, default=None,
        help="Split name under data/splits/, e.g. 'dev' or 'val' (default: dev if neither flag is given).",
    )
    split_group.add_argument(
        "--split-path", type=str, default=None,
        help="Explicit path to a split JSON file, for splits outside data/splits/.",
    )
 
    is_dev_group = parser.add_mutually_exclusive_group()
    is_dev_group.add_argument(
        "--is-dev", dest="is_dev", action="store_true", default=None,
        help="Force is_dev=True regardless of split name.",
    )
    is_dev_group.add_argument(
        "--no-is-dev", dest="is_dev", action="store_false",
        help="Force is_dev=False regardless of split name.",
    )
 
    parser.add_argument(
        "--db-path", type=str, default=str(DB_PATH_DEFAULT),
        help=f"SQLite db to write to (default: {DB_PATH_DEFAULT}).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Override the auto-generated run_id (default: '<split-name>-<timestamp>').",
    )
 
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama model name for the Faithfulness judge (default: {DEFAULT_JUDGE_MODEL}). "
             f"E.g. 'qwen2.5:3b-instruct' for the Task 13 fan-out plumbing test (DEC-D6-05).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Pipeline concurrency (default: {DEFAULT_MAX_WORKERS}, matching DEC-D6-04's production default). "
             f"Raise this together with --judge-model/OLLAMA_NUM_PARALLEL for the Task 13 fan-out test.",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Top-k for retrieval/recall/mrr metrics (default: {DEFAULT_K}). "
             f"DEC-D6-05 suggests a smaller k (e.g. 2) for the fan-out plumbing test.",
    )
    return parser.parse_args()

def print_summary(final_state: dict, split_name: str, n_expected: int, elapsed_seconds: float) -> None:
    rows = final_state.get("per_query_results", [])
 
    print("\n=== timing ===")
    print(f"elapsed: {elapsed_seconds:.2f}s ({elapsed_seconds / max(n_expected, 1):.2f}s/row avg)")
 
    print("\n=== run_status ===")
    print(f"run_status: {final_state.get('run_status')}")
    print(f"run_status_detail: {final_state.get('run_status_detail')}")
 
    print(f"\n=== per_query_results: {len(rows)} row(s) (expected {n_expected}) ===")
    # str(mode) rather than repr() -- FailureMode/QueryStatus are str,Enum
    # with a __str__ override (DEC-D6-11); repr() would print the raw
    # <FailureMode.PASS: 'pass'> form instead of the clean value.
    failure_modes = Counter(str(row.get("failure_mode")) for row in rows)
    for mode, count in sorted(failure_modes.items(), key=lambda kv: (kv[0] == "None", kv[0])):
        print(f"  {mode}: {count}")
 
    print(f"\n=== aggregated_metrics ({split_name}) ===")
    for name, value in (final_state.get("aggregated_metrics") or {}).items():
        print(f"  {name}: {value}")


def main() -> None:
    args = parse_args()
 
    split_file, split_name = resolve_split(args.split, args.split_path)
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file} "
            f"(pass --split <name under data/splits/> or --split-path <explicit path>)"
        )
 
    qa_pairs = load_split(split_file)
    print(f"Loaded split '{split_name}' from {split_file}: {len(qa_pairs)} QA pairs")
 
    is_dev = args.is_dev if args.is_dev is not None else (split_name == "dev")
    run_id = args.run_id or f"{split_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    db_path = Path(args.db_path)
 
    print(f"is_dev={is_dev}  run_id={run_id}  db_path={db_path}  "
          f"judge={args.judge_model}  k={args.k}  max_workers={args.max_workers}")
 
    rag_system = BuiltinRAGSystem(ChunkerConfig(chunk_size=500, overlap=50))
    judge_llm = InstructorStructuredLLM(args.judge_model, max_retries=3)
    metrics = [
        RecallAtK(k=args.k),
        MRR(k=args.k),
        TokenPRF1(),
        Faithfulness(llm=judge_llm),
    ]
 
    pipeline = Pipeline(
        rag_system=rag_system, metrics=metrics, k=args.k,
        max_workers=args.max_workers, db_path=db_path,
    )
 
    print(f"\n=== Running pipeline: {len(qa_pairs)} rows ===\n")
    start = time.monotonic()
    final_state = pipeline.run(qa_pairs=qa_pairs, run_id=run_id, is_dev=is_dev)
    elapsed = time.monotonic() - start
 
    print_summary(final_state, split_name, n_expected=len(qa_pairs), elapsed_seconds=elapsed)
 

if __name__ == "__main__":
    main()
