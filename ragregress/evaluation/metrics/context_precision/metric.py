"""
ragregress/evaluation/metrics/context_precision/metric.py
 
Context precision, as in Ragas (with a reference answer): are the useful chunks ranked first?
"""

import math
from typing import List
 
from ragregress.evaluation.metrics.base import Metric, MetricResult
from ragregress.evaluation.metrics.context_precision.utils import CONTEXT_PRECISION_PROMPT, ContextPrecisionVerdict


class ContextPrecision(Metric):
    name = "context_precision"
    label = "Context precision"
    description = "Average precision of the retrieved chunks: useful ones ranked first score higher (LLM judge, one call per chunk)."
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
            return MetricResult(value=math.nan, reason="No-answer question: no reference to judge the chunks against.")


        verdicts = []
        for chunk in context:
            prompt = CONTEXT_PRECISION_PROMPT.format(
                question=question, reference=reference, context=f"[{chunk.source_passage_id}]\n{chunk.text}")
            verdicts.append(self.llm.generate_structured(prompt, ContextPrecisionVerdict))


        # Calculate average precision

        useful = [bool(v.verdict) for v in verdicts]
        n_useful = sum(useful)

        score, hits = 0.0, 0
        for rank, is_useful in enumerate(useful, start=1):
            if is_useful:
                hits += 1
                score += hits / rank                   # precision@rank
        score = score / n_useful if n_useful else 0.0

        return MetricResult(
            value=score,
            details={"chunks": [
                {"passage_id": chunk.source_passage_id, "verdict": bool(v.verdict), "reason": v.reason}
                for chunk, v in zip(context, verdicts)
            ]},
        )