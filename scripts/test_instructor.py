"""
scripts/test_instructor.py

One-off empirical test, NOT wired into ragregress yet -- isolates the
exact failing row (Newton/gravity, dev split) to test whether
`instructor` + Ollama's TOOLS mode behaves differently than
LangChain's `with_structured_output(method="json_schema")` on the
NLIStatementOutput nested-list schema.

Tests two variants:
  1. No validator on NLIStatementOutput.statements (matches RAGAS's
     actual schema exactly -- see whether TOOLS mode alone fixes it).
  2. min_length=1 validator added -- see whether instructor's
     reask-on-validation-error loop recovers when TOOLS mode alone
     still comes back empty.

Requires: pip install instructor
"""

from typing import List

import instructor
from pydantic import BaseModel, Field, field_validator

# -- exact same schema shape as ragregress/evaluation/metrics/faithfulness/utils.py --

class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word") 
    # reason: str = Field(..., description="the reason for the verdict")
    # verdict: int = Field(..., description="the verdict (0/1) of the faithfulness")
    # statement: str
    reason: str
    verdict: int = Field(..., description="the verdict (0/1) of the faithfulness")


class NLIStatementOutput(BaseModel):
    """Variant 1 -- no validator, matches RAGAS's actual schema exactly."""
    statements: List[StatementFaithfulnessAnswer]


class NLIStatementOutputValidated(BaseModel):
    """Variant 2 -- min_length=1, so an empty list is a real validation
    failure instructor's reask loop can catch and retry against."""
    statements: List[StatementFaithfulnessAnswer] = Field(..., min_length=1)


NLI_STATEMENT_PROMPT = """Your task is to judge the faithfulness of a series \
of statements based on a given context. For each statement, return verdict \
as 1 if it can be directly inferred from the context, or 0 if it cannot.

Context:
{context}

Statements:
{statements}
"""

# -- exact same row that failed in the smoke test (dev qa_id 573786b51c4567190057448d) --

CONTEXT_CHUNKS = [
    ("p0002047", "Newton came to realize that the effects of gravity might be observed in different ways at larger distances. In particular, Newton determined that the acceleration of the Moon around the Earth could be ascribed to the same force of gravity if the acceleration due to gravity decreased as an inverse square law. Further, Newton realized that the acceleration due to gravity is proportional to the mass of the attracting body. Combining these ideas gives a formula that relates the mass () and the radius."),
    ("p0002049", "correction, which could account for the discrepancy. This was the first time that Newton's Theory of Gravity had been shown to be less correct than an alternative."),
    ("p0002049", "It was only the orbit of the planet Mercury that Newton's Law of Gravitation seemed not to fully explain. Some astrophysicists predicted the existence of another planet (Vulcan) that would explain the discrepancies; however, despite some early indications, no such planet could be found. When Albert Einstein formulated his theory of general relativity (GR) he turned his attention to the problem of Mercury's orbit and found that his theory added a correction, which could account for the discrepancy."),
    ("p0002046", "What we now call gravity was not identified as a universal force until the work of Isaac Newton. Before Newton, the tendency for objects to fall towards the Earth was not understood to be related to the motions of celestial objects. Galileo was instrumental in describing the characteristics of falling objects by determining that the acceleration of every object in free-fall was constant and independent of the mass of the object. Today, this acceleration due to gravity towards the surface of the Earth is called g."),
    ("p0002048", "the Earth's mass given the above equation. Newton, however, realized that since all celestial bodies followed the same laws of motion, his law of gravity had to be universal. Succinctly stated, Newton's Law of Gravitation states that the force on a spherical object of mass due to the gravitational pull of mass is."),
]

STATEMENTS = [
    "According to Newton, the effects of gravity might be observed differently at larger distances.",
    "Newton determined that the acceleration of the Moon around the Earth could be ascribed to the same force of gravity.",
    "Newton concluded that the acceleration due to gravity decreased as an inverse square law.",
]

JUDGE_MODEL = "qwen2.5:7b-instruct"


def build_prompt() -> str:
    context_txt = "\n\n".join(f"[{pid}]\n{text}" for pid, text in CONTEXT_CHUNKS)
    statements_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(STATEMENTS))
    return NLI_STATEMENT_PROMPT.format(context=context_txt, statements=statements_txt)


def main() -> None:
    prompt = build_prompt()
    client = instructor.from_provider(f"ollama/{JUDGE_MODEL}")

    print("=== Variant 1: no validator (matches RAGAS's schema exactly) ===")
    try:
        result_1 = client.chat.completions.create(
            response_model=NLIStatementOutput,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_retries=1,
        )
        print(result_1)
    except Exception as e:
        print(f"[EXCEPTION] {type(e).__name__}: {e}")

    print("\n=== Variant 2: min_length=1 + max_retries=3 (reask-on-failure) ===")
    try:
        result_2 = client.chat.completions.create(
            response_model=NLIStatementOutputValidated,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_retries=3,
        )
        print(result_2)
    except Exception as e:
        print(f"[EXCEPTION] {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("start")
    main()