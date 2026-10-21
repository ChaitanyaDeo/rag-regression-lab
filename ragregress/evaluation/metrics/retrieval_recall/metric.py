
from typing import List, Set

from ragregress.evaluation.metrics.base import Metric, MetricResult


class RecallAtK(Metric):
    """
    calculat Recall of all relevant documents/chunks,
    how may were retrived in topK

        Recall@k = (#Relevant Retrieved in topk)/(#total relevant doc)

    parameters:
    """

    name = "recall_at_k"
    required_inputs = frozenset({"retrieved_ids", "gold_ids"})
    regression_threshold = 0.05
    higher_is_better = True

    def __init__(self, k:int):
        super().__init__()
        self.k = k

    def score(self, *, retrieved_ids:List[str], gold_ids: List[str])->MetricResult:
        """
            (# gold ids retrieved in top-k) / (# total gold ids)

            Slicing to k happens here, not at call sites -- this is the one place
            that owns what "k" means for this metric. Reduces to a 0/1 hit
            indicator when gold_ids has exactly one element, which is the only
            case: data with  singular gold_passage_id can produce
        
        Args:
            retrieved_ids: List[str] of ids of documents retrieved by RAG.
            gold_ids: List[str] of ground truth ids of document.
        """

        gold_set = set(gold_ids)

        if not gold_set:
            return MetricResult(value=0.0)

        relevant_retrieved = len(set(retrieved_ids[:self.k]) & gold_set)

        return MetricResult(value = relevant_retrieved/len(gold_set))