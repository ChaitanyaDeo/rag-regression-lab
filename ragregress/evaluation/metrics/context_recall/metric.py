"""
ragregress/evaluation/metrics/context_recall/metric.py
 
Context recall, as in Ragas: how much of the reference answer can be attributed to the retrieved context.
"""
 
import math
from typing import List
 
from ragregress.evaluation.metrics.base import Metric, MetricResult
from ragregress.evaluation.metrics.context_recall.utils import (
    CONTEXT_RECALL_PROMPT, ContextRecallOutput)


class ContextRecall(Metric):
    name = "context_recall"
    label = "Context recall"
    description = "Share of the gold answer(s) that the retrieved context supports (LLM judge)."
    group = "retrieval"

    default_enabled = False

    required_context = frozenset({"llm"})

    required_inputs = frozenset({"question", "reference", "context"})

    regression_threshold = None
    higher_is_better = True

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def score(self, *, question: str, reference: str, context: List) -> MetricResult:
        if not question:
            raise ValueError("question is missing. Please add question.")
        if not isinstance(reference, str):
            raise ValueError("reference must be a string (the reference text; '' for a no-answer question)")
        if not context:
            raise ValueError("context is missing. Please add context.")

        if not reference.strip():
            return MetricResult(value=math.nan, reason="No-answer question: no reference to recall.")


        context_txt = "\n\n".join(f"[{chunk.source_passage_id}]\n{chunk.text}" for chunk in context)
        prompt = CONTEXT_RECALL_PROMPT.format(question=question, context=context_txt, answer=reference)
        result = self.llm.generate_structured(prompt, ContextRecallOutput)

        items = result.classifications
        supported = sum(1 for item in items if item.attributed)

        return MetricResult(value=supported / len(items),
            details={"statements": [
                {"statement": item.statement, "verdict": bool(item.attributed), "reason": item.reason}
                for item in items
            ]},
        )