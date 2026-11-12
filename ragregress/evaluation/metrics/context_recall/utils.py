"""
ragregress/evaluation/metrics/context_recall/utils.py
 
Structured output, prompt and reference building for ContextRecall.
Statements = the sentences of the reference answer (here: the gold answers), as in Ragas.
"""
 
from typing import List
 
from pydantic import BaseModel, Field
 
from ragregress.evaluation.metrics.token_prf1.utils import normalize_text


class ContextRecallClassification(BaseModel):
    statement: str = Field(..., description="a sentence of the reference answer, word-by-word")
    reason: str
    attributed: int = Field(..., ge=0, le=1, description="1 if the context supports the sentence, else 0")

class ContextRecallOutput(BaseModel):
    classifications: List[ContextRecallClassification] = Field(..., min_length=1)


CONTEXT_RECALL_PROMPT = """
Given a context and a answer, analyze each statement/sentence in reference answer can be attributed to given context or Not.

Rules:
1. Use only binary classification: 1 if statement can be attributed to the context, 0 if it cannot.
2. Use only context. Do not use external knowledge
3. Judge every sentence of the reference Answer, in order, one result per sentence.
4. Copy each sentence word-by-word into "statement" and give a short "reason".

Question:
{question}

Context:
{context}
 
Answer:
{answer}
"""