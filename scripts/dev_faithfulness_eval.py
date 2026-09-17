import json
from pathlib import Path
from typing import List

from langchain_ollama import ChatOllama

from ragregress import evaluate
from ragregress.benchmark.schemas import QAPair
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.evaluation.metrics import RecallAtK, MRR, TokenPRF1, Faithfulness
from ragregress.rag.no_rag_baseline import NoRAGBaseline
from ragregress.evaluation.structured_llm import InstructorStructuredLLM

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
OUT_DIR = Path("results")
K = 5
JUDGE_MODEL = "qwen2.5:7b-instruct"

def load_split(name: str) -> List[QAPair]:
    raw = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]
 
 
def print_summary(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  rows: {len(result['rows'])}")
    print(f"  aggregates: {result['aggregates']}")

def dump_faithfulness_detail(result: dict, qa_pairs: List[QAPair], path: Path) -> None:

    by_id = {qa.qa_id: qa for qa in qa_pairs}
    rows_out = []
    for row in result["rows"]:
        qa = by_id.get(row["qa_id"])
        faithfulness = row.get("faithfulness")
        rows_out.append({
            "qa_id": row["qa_id"],
            "question": qa.question if qa else None,
            "gold_answers": qa.answer if qa else None,
            "faithfulness_score": faithfulness.value if faithfulness else None,
            "faithfulness_reason": faithfulness.reason if faithfulness else None,
            "faithfulness_details": faithfulness.details if faithfulness else None,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows_out, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote per-row detail -> {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_split = load_split("dev")
    print(f"Loaded dev split: {len(dev_split)} QA pairs")
 
    judge_llm = InstructorStructuredLLM(model_name=JUDGE_MODEL, max_retries=5)
 
    # 1. BuiltinRAGSystem + all 4 metrics, full split.
    metrics_with_faithfulness = [
        RecallAtK(k=K),
        MRR(k=K),
        TokenPRF1(),
        Faithfulness(llm=judge_llm),
    ]
    builtin = BuiltinRAGSystem(chunker_config=ChunkerConfig(chunk_size=500, overlap=50))
    builtin_result = evaluate(builtin, dev_split, metrics_with_faithfulness, k=K)
    print_summary("BuiltinRAGSystem (full dev split)", builtin_result)
    dump_faithfulness_detail(builtin_result, dev_split, OUT_DIR / "dev_faithfulness_detail.json")



    no_rag = NoRAGBaseline()
    metrics_no_rag = [RecallAtK(k=K), MRR(k=K), TokenPRF1()]
    no_rag_result = evaluate(no_rag, dev_split, metrics_no_rag, k=K)
    print_summary("NoRAGBaseline (no Faithfulness, full dev split)", no_rag_result)


    by_id = {qa.qa_id: qa for qa in dev_split}
    sample_rows = no_rag_result["rows"][:5]
    print("\n  token_prf1 spot-check (first 5 rows):")
    for row in sample_rows:
        qa = by_id.get(row["qa_id"])
        tprf1 = row["token_prf1"]
        chunks = no_rag.retrieve(qa.question)
        generated_answer = no_rag.generate(qa.question, chunks)
        print(f"    [{row['qa_id']}] token_prf1={tprf1.value!r}")
        print(f"      gold:      {qa.answer!r}")
        print(f"      generated: {generated_answer!r}")


if __name__ == "__main__":
    main()
