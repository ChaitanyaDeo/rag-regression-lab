"""
ragregress/evaluation/metrics/retrieval_mrr/metric.py
"""

from typing import List, Set

from ragregress.evaluation.metrics.base import Metric, MetricResult

class MRR(Metric):
    name = "mrr"
    required_inputs = frozenset({"retrieved_ids", "gold_ids"})

    regression_threshold = 0.05
    higher_is_better = True

    label = "MRR"
    description = "Reciprocal rank of the first gold passage in the top-k."
    group = "retrieval"
    default_enabled = True
    required_context = frozenset({"k"})

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def score(self, *, retrieved_ids: List[str], gold_ids: List[str]) -> MetricResult:
        """
            1 / (1-based rank of the first gold id found) within the top-k of
            retrieved_ids, 0.0 if none of gold_ids appear in the top-k

            Standard MRR definition: rank of the first relevant item, regardless
            of how many relevant items exist in total. Capped at k for the same
            reason as recall_at_k -- keeps this correct regardless of how many
            candidates the caller's retriever actually returns.
        """

        gold_set = set(gold_ids)
        for idx, rid in enumerate(retrieved_ids[:self.k]):
            if rid in gold_set:
                return MetricResult(value=1.0 / (idx + 1))
        return MetricResult(value=0.0)


    