"""
ragregress/evaluation/reference.py

The gold answers of a question as one reference text, for metrics that judge a whole reference
(the pipeline passes it to them as `reference`).
"""

from typing import Iterable

from ragregress.evaluation.metrics.token_prf1.utils import normalize_text


def build_reference(gold_answers: Iterable[str]) -> str:
    """One sentence per distinct gold answer, one per line, in order.

    Distinct = after SQuAD normalization (case, punctuation, articles, spacing). Empty golds are dropped,
    so a no-answer question (golds == ['']) gives ''. A gold without end punctuation gets a full stop,
    so a judge that splits the reference into sentences sees one sentence per gold.
    """
    seen, sentences = set(), []
    for gold in gold_answers or []:
        text = (gold or "").strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        sentences.append(text if text[-1] in ".!?" else text + ".")
    return "\n".join(sentences)