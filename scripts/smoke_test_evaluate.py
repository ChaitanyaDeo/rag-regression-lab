
"""
A smoke test to check implementaion of evaluate()

not full eval run, to check single pass retrive and generate,
row shape, aggregrate etc.


"""

import json
from pathlib import Path
from typing import List

from langchain_ollama import ChatOllama

from ragregress import evaluate
from ragregress.benchmark.schemas import QAPair
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.evaluation.structured_llm import LangChainStructuredLLM
from ragregress.evaluation.metrics import RecallAtK, MRR, TokenPRF1, Faithfulness
from ragregress.rag.no_rag_baseline import NoRAGBaseline
from ragregress.evaluation.structured_llm import InstructorStructuredLLM

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
K = 5
N_SAMPLE = 3

JUDGE_MODEL = "qwen2.5:7b-instruct"


def load_split(name: str) -> List[QAPair]:
    """loader.py has no split-loading helper -- read the saved split JSON
    directly, same pattern as verify_day2_ingest.py's load_passages().
    """
    raw = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [QAPair(**row) for row in raw]



def print_result(label: str, result: dict) -> None:
    print(f"\n=== {label} ===\n")
    for row in result["rows"]:
        print(f"  [{row['qa_id']}]")
        for metric_name, metric_result in row.items():
            if metric_name == "qa_id":
                continue
            print(f"    {metric_name}: value={metric_result.value!r} reason={metric_result.reason!r}")
    print(f"\n  aggregates: {result['aggregates']}")


def main() -> None:
    sample = load_split("dev")[:N_SAMPLE]

    # judge_llm = LangChainStructuredLLM(ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=8192))

    judge_llm = InstructorStructuredLLM(JUDGE_MODEL, max_retries=3)

    metrics = [
        RecallAtK(k=K),
        MRR(k=K),
        TokenPRF1(),
        Faithfulness(llm=judge_llm),
    ]

    # 1. BuiltInRAGSystem -- our actual RAG that exercises retrival and generation

    builtin = BuiltinRAGSystem(ChunkerConfig(chunk_size=500, overlap=50))

    builtin_results = evaluate(rag_system=builtin, qa_pairs=sample, metrics=metrics)

    print_result("BuiltInRAGsystem", builtin_results)


    # 2a. NoRAGBaseline + Faithfulness
    no_rag = NoRAGBaseline()
    try:
        evaluate(no_rag, sample, metrics, k=K)
        print("\n=== NoRAGBaseline + Faithfulness: UNEXPECTED -- did not raise ===")
    except ValueError as e:
        print(f"\n=== NoRAGBaseline + Faithfulness: raised as expected -- {e} ===")
 
    # 2b. NoRAGBaseline WITHOUT Faithfulness -- the combination that's
    metrics_without_faithfulness = [RecallAtK(k=K), MRR(k=K), TokenPRF1()]
    no_rag_result = evaluate(no_rag, sample, metrics_without_faithfulness, k=K)
    print_result("NoRAGBaseline (no Faithfulness)", no_rag_result)


if __name__ == "__main__":
    main()
