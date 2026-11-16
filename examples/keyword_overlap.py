"""
An example for custom pipeline rag
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Set

from dotenv import find_dotenv, load_dotenv
from pydantic import Field

from ragregress.core.config import RunConfig
from ragregress.core.runner import execute
from ragregress.evaluation.comparator import compare_runs
from ragregress.evaluation.regression import assess_regression
from ragregress.rag.interface import Chunk, GenerationResult, RAGSystem
from ragregress.rag.registry import PIPELINE_REGISTRY, PipelineEntry, PipelineParams
from ragregress.report import default_output_path, generate_report, write_report
from ragregress.storage.repositories import Repository

PIPELINE_ID = "keyword_overlap"
TARGETS = Path("configs/targets/squad_val.yaml")
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------- 1. the pipeline

class KeywordOverlapRAG:
    """Retrieves the passages sharing the most words with the question; answers with the
    best-matching sentence of the top passage. No embeddings, no LLM, deterministic."""

    def __init__(self, passages_path: str = "data/passages.json", min_word_len: int = 3):
        self.min_word_len = min_word_len
        raw = json.loads(Path(passages_path).read_text(encoding="utf-8"))
        self._passages = [(p["passage_id"], p["text"], self._terms(p["text"])) for p in raw]

    def _terms(self, text: str) -> Set[str]:
        return {w for w in _WORD.findall(text.lower()) if len(w) >= self.min_word_len}

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        q = self._terms(query)
        top = sorted(self._passages, key=lambda p: (-len(q & p[2]), p[0]))[:k]
        return [Chunk(chunk_id=f"{pid}::0", source_passage_id=pid, text=text) for pid, text, _ in top]

    def generate(self, query: str, contexts: List[Chunk]) -> GenerationResult:
        if not contexts:
            return GenerationResult(raw_output="", answer="")
        q = self._terms(query)
        sentences = _SENTENCE_END.split(contexts[0].text)
        best = max(sentences, key=lambda s: (len(q & self._terms(s)), -len(s)))
        return GenerationResult(raw_output=best, answer=best, citations=[contexts[0].source_passage_id])


# ---------------------------------------------------------------- 2. registration (this process only)

class KeywordOverlapParams(PipelineParams):
    """The pipeline's knobs. Every field is part of config_hash; unknown keys are rejected."""
    min_word_len: int = Field(default=3, ge=1, le=20)


def build_keyword_overlap(params: KeywordOverlapParams) -> RAGSystem:
    return KeywordOverlapRAG(min_word_len=params.min_word_len)


PIPELINE_REGISTRY[PIPELINE_ID] = PipelineEntry(KeywordOverlapParams, build_keyword_overlap)


# ---------------------------------------------------------------- 3. config, run, compare, report

def make_config(split: str, min_word_len: int) -> RunConfig:
    return RunConfig.model_validate({
        "pipeline": {"id": PIPELINE_ID, "params": {"min_word_len": min_word_len}},
        "top_k": 5,
        "judge_model": "qwen2.5:7b-instruct",
        "judge_seed": 42,
        "eval_split": split,
        "is_dev": split == "dev",
    })


def run_once(cfg: RunConfig) -> str:
    outcome = execute(cfg)                                   # same engine as `ragregress run`
    rows = outcome.final_state["per_query_results"]
    print(f"run_id {outcome.run_id}  hash {outcome.config_hash[:12]}  {outcome.elapsed_s:.0f}s  log {outcome.log_path}")
    print("  failure modes:", dict(Counter(str(r.get("failure_mode")) for r in rows)))
    for name, value in outcome.final_state["aggregated_metrics"].items():
        if not name.startswith("latency_"):
            print(f"  {name:<13} {value:.4f}")
    return outcome.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "val", "test"], default="dev")
    split = parser.parse_args().split
    load_dotenv(find_dotenv(usecwd=True))                    # OLLAMA_HOST, JUDGE_* (as the CLI does)

    baseline_id = run_once(make_config(split, min_word_len=3))
    if split == "dev":
        print("dev runs cannot be compared; use --split val for the comparison and report")
        return 0
    candidate_id = run_once(make_config(split, min_word_len=4))

    repo = Repository(Path("ragregress.db"))
    report = assess_regression(compare_runs(repo, baseline_id, candidate_id))
    print(f"\nVERDICT {report.verdict.value} (exit code {report.exit_code})")
    for m in report.metrics:
        print(f"  {m.name:<13} {m.before} -> {m.after}  delta {m.delta}  {m.status.value}")
    for reason in report.reasons:
        print("  -", reason)

    ctx = generate_report(repo, baseline_id, candidate_id, targets_path=TARGETS if TARGETS.is_file() else None)
    print("report:", write_report(ctx, default_output_path(baseline_id, candidate_id)).resolve())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())