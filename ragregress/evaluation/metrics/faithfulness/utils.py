"""
    ragregress/evaluation/metrics/faithfulness/utils.py
    
    Statements / claims - Basically atomic facts extracted from answer
"""

from typing import List, Dict

from pydantic import BaseModel, Field

class StatementGeneratorOutput(BaseModel):
    """Structured output for statement(claims) generation"""

    statements: List[str] = Field(..., description="Atomic statements extracted from the answer")

class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str
    verdict: int = Field(..., description="the verdict (0/1) of the faithfulness")


class NLIStatementOutput(BaseModel):
    statements: List[StatementFaithfulnessAnswer] = Field(..., min_length=1)


STATEMENT_GENERATOR_PROMPT = """You are a statement decomposition system.

Given a Question and an Answer, decompose every sentence in the Answer into
one or more simple, self-contained statements.

Rules:
1. Cover all factual information from the Answer.
2. Split complex sentences into multiple statements when necessary.
3. Each statement must express exactly one clear fact.
4. Every statement must be understandable without the Question or other statements.
5. Do NOT use pronouns such as: he, she, it, they, them, this, that, these, those,
   his, her, their, its, etc.
6. Replace pronouns with the specific person, object, place, or concept being
   referenced.
7. Do not add information that is not present in the Answer.
8. Do not remove important information from the Answer.
9. Keep the wording concise.
10. Output only the statements, with one statement per line.
11. Do not number the statements.
12. Do not provide explanations or commentary.

Question:
{question}

Answer:
{answer}

Statements:
"""
# another prompt that can be used:
# STATEMENT_GENERATOR_PROMPT = """Given a question and an answer, analyze the \
# complexity of each sentence in the answer. Break down each sentence into \
# one or more fully understandable statements. Ensure that no pronouns are \
# used in any statement.

# Question: {question}
# Answer: {answer}
# """

# NLI_STATEMENT_PROMPT = """You are a Natural Language Inference (NLI) classifier.

# Your task is to determine whether each statement is supported by the given
# context.

# Classification:
# - Return 1 if the statement is directly supported or can be logically inferred
#   from the context.
# - Return 0 if the statement is not supported by the context, contradicts the
#   context, or requires information that is not present in the context.

# Rules:
# 1. Judge each statement independently.
# 2. Use only information from the context.
# 3. Do not use outside knowledge.
# 4. Do not assume facts that are not stated or logically implied.
# 5. A statement can receive 1 even if the exact wording does not appear in the
#    context, as long as the meaning is clearly supported.
# 6. If any important part of a statement is unsupported, return 0.
# 7. Return exactly one verdict (1 or 0) for each statement.
# 8. Preserve the order of the statements.
# 9. Output only the verdicts separated by commas.
# 10. Do not provide explanations.

# Context:
# {context}

# Statements:
# {statements}

# Verdicts:
# """


#Another prompt for NLI verdicts:
NLI_STATEMENT_PROMPT = """Your task is to judge the faithfulness of a series \
of statements based on a given context. For each statement, return verdict \
as 1 if it can be directly inferred from the context, or 0 if it cannot.

Context:
{context}

Statements:
{statements}
"""

