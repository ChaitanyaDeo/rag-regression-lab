
import re
from typing import List, NamedTuple
import string

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