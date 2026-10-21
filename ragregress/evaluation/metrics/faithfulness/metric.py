"""
Faithfulness Metric - Measure whether generated text is supported by
    retrieved context

    faithfulness = (#supported claims)/(#total claims)

    It is a non determinisitic, generator metric

    Ragas inspired implementaion
"""

import math

from typing import List

from ragregress.evaluation.metrics.base import Metric, MetricResult
from ragregress.evaluation.structured_llm import StructuredLLM
from ragregress.rag.interface import Chunk
from .utils import (
    StatementGeneratorOutput,
    NLIStatementOutput,

    STATEMENT_GENERATOR_PROMPT,
    NLI_STATEMENT_PROMPT,
)

class Faithfulness(Metric):
    name = "faithfulness"
    required_inputs = frozenset({"question", "answer", "context"})
    regression_threshold = 0.05
    higher_is_better = True

    def __init__(self, llm:StructuredLLM):
        super().__init__()
        self.llm = llm


    def score(self, *, question:str, answer:str, context:List[Chunk])-> MetricResult:
        if not question:
            raise ValueError("question is missing, Please add a quesition")
        if not answer:
            raise ValueError("answer is missing. Please add answer")
        if not context:
            raise ValueError("context is missing. Please add context.")
        
        # 1: extract claims
        statements = self._extract_claims(question, answer,) # Claims made by answer

        if not statements:
            return MetricResult(value=math.nan, reason="No claims extracted.")

        # 2: Call judge to evaluate the statements

        context_txt ="\n\n".join(f"[{chunk.source_passage_id}]\n{chunk.text}" for chunk in context)
        verdicts = self._judge_claims(statements, context_txt)

        if len(verdicts.statements) != len(statements):
            return MetricResult(
                value=math.nan,
                reason=f"judge returned {len(verdicts.statements)} verdicts for {len(statements)} statements",
            )

        # 3: Calculate faithfulness
        score = self._compute_score(verdicts)

        

        return MetricResult(value=score,
                            details={
                                "statements": [
                                        {
                                            "statement": s.statement,
                                            "verdict": bool(s.verdict),
                                            "reason": s.reason,
                                        }
                                        for s in verdicts.statements
                                    ]
                                },
                            )

    def _extract_claims(self, questions: str, answer:str)-> List[str]:
        prompt = STATEMENT_GENERATOR_PROMPT.format(question=questions, answer=answer)

        result = self.llm.generate_structured(prompt, StatementGeneratorOutput)

        return result.statements

    def _judge_claims(self, statements: List[str], context:str)->NLIStatementOutput:
        prompt = NLI_STATEMENT_PROMPT.format(context=context, 
                                             statements = "\n".join(f"{i+1}. {s}" for i,s in enumerate(statements)))

        result = self.llm.generate_structured(prompt, NLIStatementOutput)

        return result

    def _compute_score(self, verdicts: NLIStatementOutput) -> float:

        faithful_statements = sum(1 if statement.verdict else 0 for statement in verdicts.statements)

        score = faithful_statements/len(verdicts.statements)

        return score
    