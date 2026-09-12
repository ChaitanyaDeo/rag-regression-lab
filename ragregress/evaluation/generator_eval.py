
"""
Generator stage metrics: token-level f1

More evals to be added
"""

import re
import string
from collections import Counter
from typing import List, Dict, NamedTuple
from ragregress.rag.interface import RAGSystem
from ragregress.benchmark.schemas import QAPair

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

class PRF1(NamedTuple):
    """Precision/recall/F1 for one prediction-vs-gold-answer comparison.
 
    Exists so precision and recall (not just the final F1) are available
    to callers that want to build a P-R curve — e.g. across a sweep of
    predictions/thresholds — without recomputing token overlap themselves.
    """
 
    precision: float
    recall: float
    f1: float

def normalize_text(text:str) -> str:
    """ Normalize text as:
    Lower text and remove punctuation, articles and extra whitespace.

    This is SQuAD style
    """

    def remove_article(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuations(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_article(remove_punctuations(lower(text))))


def get_tokens(s: str)-> List[str]:
    """Normalize and whitespace-tokenize a string."""
    if not s: 
        return []

    return normalize_text(s).split()


def compute_token_level_f1(ans_generated: str, ans_gt: str) -> PRF1:
    """
    Token-level F1 between one prediction and one gold answer
    
    Handles the SQuAD v2-style empty-answer edge case: if either
    normalizes to zero tokens, F1 is 1.0 when both are empty 
    
    this also guards the precision/recall division from a ZeroDivisionError.
    """

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

def answer_precision_recall_f1(prediction: str, gold_answers: List[str]) -> PRF1:
    """Precision/recall/F1 between a prediction and the best-matching gold answer.
 
    gold_answers is assumed non-empty, per QAPair.answer's schema
    guarantee — an empty list is a data/schema violation, not a valid
    "no answer" case (that's represented as [""], handled by
    compute_token_level_f1's empty-token check above), so it raises rather
    than silently returning an arbitrary score.
    """
 
    if not gold_answers:
        raise ValueError(
            "answer_precision_recall_f1: gold_answers must be non-empty "
            "(QAPair.answer is never []; use [''] to represent a genuine "
            "no-answer case)."
        )
 
    return max((compute_token_level_f1(prediction, gold) for gold in gold_answers),key=lambda prf1: prf1.f1,)

def answer_f1(ans_generated: str, gold_answers: List[str]) -> float:
    """
    Thin wrapper over answer_precision_recall_f1 for callers that only need
    the scalar F1 (e.g. existing hand-tests, aggregate-mean reporting)

    SQuAD-standard convention is max F1 across all accepted
    answers for the question, not average — a prediction shouldn't be
    penalized for not matching every acceptable phrasing when it matches
    at least one.

    Returns:
        f1 score : max across all gold answers
    """

    return answer_precision_recall_f1(ans_generated, gold_answers).f1



