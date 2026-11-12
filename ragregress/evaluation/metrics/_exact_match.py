"""
ragregress/evaluation/metrics/exact_match/metric.py
"""
from typing import List
 
from ragregress.evaluation.metrics.base import Metric, MetricResult
from ragregress.evaluation.metrics.token_prf1.utils import normalize_text


class ExactMatch(Metric):
    """
    1.0 if the generated answer equals any gold answer after SQuAD-style normalization
    (lower-case, no punctuation, no articles, collapsed whitespace), else 0.0
    
    Strict by design: a correct but longer answer scores 0.0 (TokenPRF1 gives partial credit).
    As in TokenPRF1, gold_answers must be non-empty; use [''] for a genuine no-answer case,
    which an empty (after normalization) answer matches.
 
    Reported only: no regression threshold, so it does not gate the verdict.
    """

    name = "exact_match"
    required_inputs = frozenset({"answer", "gold_answers"})
    regression_threshold = None
    higher_is_better = True

    def __init__(self):
        super().__init__()

    def score(self, *, answer: str, gold_answers: List[str]) -> MetricResult:
        if not gold_answers:
            raise ValueError(
                "ExactMatch.score: gold_answers must be non-empty "
                "(use [''] to represent a genuine no-answer case)"
            )
 
        predicted = normalize_text(answer or "")
        matched = any(predicted == normalize_text(gold) for gold in gold_answers)
        return MetricResult(value=1.0 if matched else 0.0)