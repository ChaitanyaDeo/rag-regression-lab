"""
ragregress/evaluation/metrics/token_prf1/metric.py
"""

from collections import Counter
from typing import List

from ragregress.evaluation.metrics.token_prf1.utils import PRF1, get_tokens

from ragregress.evaluation.metrics.base import Metric, MetricResult


class TokenPRF1(Metric):
    """
        Token level metric for generator, 
        comparing prediction to gold answers

        Handles the SQuAD v2-style empty-answer edge case: if either
        normalizes to zero tokens, F1 is 1.0 when both are empty
    """

    name = "token_prf1"
    required_inputs = frozenset({"answer", "gold_answers"})
    regression_threshold = 0.05
    higher_is_better = True

    label = "Token F1"
    description = "Token-overlap F1 between the answer and the best gold answer (SQuAD-style)."
    group = "answer"
    default_enabled = True

    def __init__(self):
        super().__init__()

    def score(self, *, answer: str, gold_answers: List[str])-> MetricResult:
        if not gold_answers:
            raise ValueError(
                "TokenPRF1.socre: gold_answers must be non-empty."
                "(use [''] to represent a genuine no-answer case)"
            )

        best = max((self._compute(answer, ans_gt) for ans_gt in gold_answers),
                   key = lambda prf1: prf1.f1)

        return MetricResult(value=best.f1, details={"precision": best.precision, "recall": best.recall})

    @staticmethod
    def _compute(ans_generated: str, ans_gt: str) -> PRF1:
        
        pred_tokens = get_tokens(ans_generated)
        gt_tokens = get_tokens(ans_gt)

        common = Counter(pred_tokens) & Counter(gt_tokens)

        num_same = sum(common.values())

        if len(gt_tokens) == 0 or len(pred_tokens) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            match = float(gt_tokens == pred_tokens)
            return PRF1(precision=match, recall=match, f1=match)

        if num_same == 0:
            return PRF1(precision=0.0, recall=0.0, f1=0.0)
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return PRF1(precision=precision, recall=recall, f1=f1)