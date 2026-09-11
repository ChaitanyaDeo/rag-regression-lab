"""
ragregress/benchmark/schemas.py

Pydantic models for the eval benchmark: a passage (deduplicated corpus document)

a QA pair (linked to its exact gold passage)

the CRAG - category taxonomy (user-overridable, defaults to CRAG's 8 types).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# CRAG's 8 question types (Meta, KDD Cup 2024) — the default category set.
# `category` on QAPair stays a plain str so a user can override with their
# own taxonomy; this list only documents/seeds the default. 
# The bare list of names, where needed, is just - CRAG_CATEGORY_DEFINITIONS.keys()
CRAG_CATEGORY_DEFINITIONS = {
    "simple": "a single fact lookup",
    "simple_with_condition": "a fact lookup with a qualifying condition (e.g. a date range, a specific context)",
    "set": "expects a list/set of items as the answer",
    "comparison": "compares two or more entities",
    "aggregation": "requires counting, summing, or another aggregation over multiple facts",
    "multi-hop": "requires chaining facts across more than one reasoning step",
    "post_processing_heavy": "requires non-trivial computation/formatting on retrieved facts",
    "false_premise": "the question contains an incorrect assumption",
}


class Passage(BaseModel):
    """A single deduplicated corpus document"""

    passage_id: str = Field(..., description = "Unique identifier for this passage. Stable id assigned at dedup time, e.g. 'p0000042'")
    title: str = Field(..., description = "SQuAD article title this passage came from")
    text: str = Field(...,  description = "The passage text (a SQuAD 'context' value)")


class QAPair(BaseModel):
    """A single question-answer pair, linked to its exact gold passage"""

    qa_id: str = Field(..., description = "Unique identifier for a QA pair, (Original SQuAD row id for this rag)")

    question: str

    answer: List[str] = Field(..., description = "All accepted answerr strings for QA pair")
    answer_start: List[int] = Field(..., description = "All accpeted answer start offsets for QA pair")

    gold_passage_ids: List[str] = Field(..., description="passage_id(s) of the exact source context(s) for this question.")

    title: str = Field(..., description = "article title (denormalized for convenience)")

    category: Optional[str] = Field(default = None,
                                    description = "User-defined category, defaults to CRAG's 8 types (see CRAG_CATEGORIES).")
