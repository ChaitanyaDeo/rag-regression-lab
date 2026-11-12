"""
ragregress/evaluation/metrics/_bleu.py
"""

from typing import List
 
from sacrebleu.metrics import BLEU as _SacreBLEU
 
from ragregress.evaluation.metrics.base import Metric, MetricResult


class BLEU(Metric):
    """
    Sentence-level BLEU (n-gram precision up to 4, brevity penalty) computed using `sacrebleu`
    scaled from its 0-100 to 0-1

    Case-insensitive; `effective_order=True` (sacrebleu's
    recommended setting for single sentences) so a short answer is not scored 0 only because it
    has no 4-grams.

    All gold answers are passed to sacrebleu as references of one hypothesis.


    """

    name = "bleu"
    required_inputs = frozenset({"answer", "gold_answers"})
    regression_threshold = None
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self._bleu = _SacreBLEU(effective_order=True, lowercase=True)

    def score(self, *, answer: str, gold_answers: List[str]) -> MetricResult:
        if not gold_answers:
            raise ValueError(
                "Bleu.score: gold_answers must be non-empty "
                "(use [''] to represent a genuine no-answer case)"
            )
 
        result = self._bleu.sentence_score(answer or "", list(gold_answers))
        return MetricResult(value=result.score / 100.0)