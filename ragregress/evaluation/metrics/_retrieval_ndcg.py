"""
ragregress/evaluation/metrics/retrieval_ndcg/metric.py
"""

import math
from typing import List

from ragregress.evaluation.metrics.base import Metric, MetricResult

class NDCGAtK(Metric):
    """
    Rank aware retrival metric with binary relevance(a retrival is relevant or not)

    Measures how closely the top-k retrieved results match the ideal ranking, normalized to a score between 0 and 1.

        DCG@k = sum over gold passage found at 1-based rank r (r <= k) of 1/log2(r+1) 

        IDCG@k = sum over gold passage found 

        NDCG@k = DCG@k / IDCG@k

    A passage is counted once, at its first rank; a repeated chunk of it still occupies a
    rank but adds nothing, so the value cannot exceed 1.
    """

    name = "ndcg_at_k"
    required_inputs = frozenset({"retrieved_ids", "gold_ids"})
    regression_threshold = None
    higher_is_better = True

    def __init__(self, k:int):
        super().__init__()
        if k < 1:
            raise ValueError(f"NDCGAtK: k must be >= 1, got {k}")
        self.k = k

    def score(self, * , retrieved_ids: List[str], gold_ids: List[str])-> MetricResult:
        """
        Args:
            retrieved_ids: List[str] of ids of documents retrieved by RAG.
            gold_ids: List[str] of ground truth ids of document.
        """

        gold_set = set(gold_ids)
        if not gold_set:
            return MetricResult(value=0.0, reason="No document found in gold_set")

        seen = set()
        dcg = 0.0

        for rank, rid in enumerate(retrieved_ids[:self.k], start=1):
            if rid in gold_set and rid not in seen:
                dcg += 1.0/math.log2(rank + 1)
            seen.add(rid)

        idcg = sum(1.0/math.log2(rank +1) for rank in range(1, min(len(gold_set), self.k)+1))

        if idcg ==0:
            return MetricResult(value=0.0, reason="No relevant documents found in gold set")

        return MetricResult(value = dcg/idcg)