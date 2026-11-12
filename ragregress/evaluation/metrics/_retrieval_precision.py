"""
ragregress/evaluation/metrics/retrieval_precision/metric.py
"""

from typing import List, Set

from ragregress.evaluation.metrics.base import Metric, MetricResult

class PrecisionAtK(Metric):
    """
    Calculate precision of all relevant documents/chunks,

    How much of the top-k is relevant?

        Precision@k = (#Relevant Retrieved in top-k)/k

    The denominator is always k, also when the retriever returned fewer than k chunks
    (missing slots count as misses). Two chunks of one passage count once, so the value
    stays in [0, 1] and matches how RecallAtK treats ids (as a set).
    """

    name = "precision_at_k"
    required_inputs = frozenset({"retrieved_ids", "gold_ids"})
    regression_threshold = None
    higher_is_better = True

    label = "Precision@k"
    description = "Share of the top-k slots filled by a distinct gold passage."
    group = "retrieval"
    default_enabled = False
    required_context = frozenset({"k"})

    def __init__(self, k:int):
        super().__init__()
        if k<1:
            raise ValueError(f"PrecisionAtK: k must be >= 1, got {k}")
        self.k = k


    def score(self, *, retrieved_ids:List[str], gold_ids: List[str]) -> MetricResult:
        """
            (# gold ids retrieved in top-k) / k

        Args:
            retrieved_ids: List[str] of ids of documents retrieved by RAG.
            gold_ids: List[str] of ground truth ids of document.
        """

        gold_set = set(gold_ids)

        if not gold_set:
            return MetricResult(value=0.0)

        hits = len(set(retrieved_ids[:self.k]) & gold_set)

        return MetricResult(value = hits/self.k)