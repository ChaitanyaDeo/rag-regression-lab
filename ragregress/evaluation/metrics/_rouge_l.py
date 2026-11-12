"""
ragregress/evaluation/metrics/_rouge_l.py
"""

from typing import List, Set

from rouge_score import rouge_scorer

from ragregress.evaluation.metrics.base import Metric, MetricResult

class RougeL(Metric):
    """
    Calculate Rouge-L F-measure: overlap by longest common subsequence of tokens, so word order matters.
    Computed by the `rouge-score` package with its default tokenizer (lower-case, non-alphanumerics dropped) and no stemming.

    Several gold answers: the best F-measure wins (precision / recall of that best match go to `details`).
    gold_answers must be non-empty, as in TokenPRF1 / ExactMatch.

    Library behaviour kept as is: an empty answer scores 0.0 even against an empty gold answer
    (ExactMatch and TokenPRF1 score that no-answer case 1.0).

    """

    name = "rouge_l"
    required_inputs = frozenset({"answer", "gold_answers"})
    regression_threshold = None
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self._scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    def score(self, *, answer: str, gold_answers: List[str]) -> MetricResult:
        if not gold_answers:
            raise ValueError("RougeL.score: gold_answers must be non-empty "
                             "(use [''] to represent a genuine no-answer case)")
        
        # rouge-score takes (target, prediction), in that order
        best = max((self._scorer.score(gold, answer or "")["rougeL"] for gold in gold_answers), key=lambda s: s.fmeasure,)

        return MetricResult(
            value=best.fmeasure,
            details={"precision": best.precision, "recall": best.recall},
        )
