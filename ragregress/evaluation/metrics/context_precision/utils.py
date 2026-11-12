"""
ragregress/evaluation/metrics/context_precision/utils.py
 
Structured output and prompt for ContextPrecision (one verdict per retrieved chunk, as in Ragas).
"""


from pydantic import BaseModel, Field
 
 
class ContextPrecisionVerdict(BaseModel):
    reason: str
    verdict: int = Field(..., ge=0, le=1, description="1 if the context was useful in arriving at the answer, else 0")


CONTEXT_PRECISION_PROMPT ="""You are a retrieval relevance judge.
 
Given a Question, a reference Answer and one retrieved Context chunk, decide whether the
Context chunk is useful in arriving at the reference Answer.
 
Rules:
1. Return verdict = 1 only if the Context chunk itself contains the information stated in the Answer,
   or a key fact that is needed to arrive at it.
2. Return verdict = 0 if the chunk only discusses the same topic, the same people or related background
   without that information. Being on the same subject is not enough.
3. Judge only this Context chunk. Do not use outside knowledge.
4. Give a short "reason" (one sentence).
 
Question:
{question}
 
Answer:
{reference}
 
Context:
{context}
"""