"""
ragregress/rag/preflight.py

Preflight validation for custom RAG pipelines.

One pure function: system, list of QA pairs, k, set of corpus ids → list of findings. 
A finding has: check name, severity (error or warning), message, question id.

Check->Severity->Message must say
retrieve result is not a list of Chunk          error               the actual type
more than k chunks	                            error	            got N, k is K
zero chunks                                     error           	the question used
source_passage_id not in corpus ids	            error           	the offending id and "use the passage id of the dataset, not your own chunk id"
generate result is not a GenerationResult	    error	            the actual type
empty or whitespace answer                  	error           	"faithfulness and token F1 cannot be scored"
a chunk has empty text	                        warning         	"faithfulness has no context to check against"
duplicate source_passage_id in the top k	    warning	            "only the first occurrence counts for MRR"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, List

from ragregress.rag.interface import Chunk, GenerationResult

import logging

logger = logging.getLogger(__name__)  

ERROR_STR = "error"
WARNING_STR = "warning"


@dataclass(frozen=True)
class PreflightFinding:
    check: str
    severity: str
    message: str
    question_id: str

@dataclass(frozen=True)
class PreflightQuestionResult:
    question_id: str
    status: str
    exception: Optional[str] = None


@dataclass
class PreflightResult:
    findings: list[PreflightFinding] = field(default_factory=list)
    questions: list[PreflightQuestionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(
            finding.severity == ERROR_STR
            for finding in self.findings
        )



def run_preflight(system, qa_pairs: List, k:int, corpus_ids: Optional[Set[str]]=None) -> PreflightResult:
    """Validate retrieval and generation behavior without logging."""

    result = PreflightResult()

    def add_finding(question_id, check, severity, message):
        result.findings.append(
            PreflightFinding(check=check, severity=severity, message=message, question_id=question_id)
        )

    for qa in qa_pairs:
        question_id = str(qa.qa_id)
        question = qa.question

        try:
            chunks = system.retrieve(question,k)
        except Exception as exc:
            result.questions.append(PreflightQuestionResult(
                    question_id=question_id,
                    status="raised",
                    exception=f"{type(exc).__name__}: {exc}",
                ))
            continue

        # Retrieval return type.
        if not isinstance(chunks, list):
            add_finding(question_id=question_id, check="retrieve_return_type", severity=ERROR_STR,
                        message="retrieve() must return a list of Chunk objects.")

            result.questions.append(
                PreflightQuestionResult(question_id, "invalid")
            )
            continue

        # Retrieval count.
        if len(chunks)>k:
            add_finding(question_id=question_id, check="too_many_chunks", severity=ERROR_STR,
                        message=f"retrieve() returned {len(chunks)} chunks; maximum is {k}.")

        if not chunks:
            add_finding(question_id=question_id, check="zero_chunks", severity=ERROR_STR,
                        message="retrieve() returned no chunks")   

            result.questions.append(
                PreflightQuestionResult(question_id, "invalid")
            )
            continue

        # Validate individual chunks.
        
        invalid_chunks = False

        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                add_finding(question_id, "chunk_type", ERROR_STR,
                    "retrieve() returned an item that is not a Chunk.",
                )
                invalid_chunks = True
                continue

            if corpus_ids is not None and (chunk.source_passage_id not in corpus_ids):
                add_finding(question_id, "unknown_passage_id", ERROR_STR,
                    f"Unknown source_passage_id: {chunk.source_passage_id!r}.",
                )

            if not chunk.text or not chunk.text.strip():
                add_finding(question_id, "empty_chunk_text", WARNING_STR,
                    f"Chunk {chunk.chunk_id!r} has empty text.",
                )

        # Duplicate source passage IDs.
        source_ids = [chunk.source_passage_id for chunk in chunks if isinstance(chunk, Chunk)]

        if len(source_ids) != len(set(source_ids)):
            add_finding(question_id, "duplicate_source_passage_id", WARNING_STR,
                "Multiple chunks reference the same source passage."
            )

        if invalid_chunks:
            result.questions.append(PreflightQuestionResult(question_id, "invalid"))
            continue

        # Generation
        try:
            generated = system.generate(question, chunks)
        except Exception as exc:
            result.questions.append(
                PreflightQuestionResult(
                    question_id=question_id,
                    status="raised",
                    exception=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if not isinstance(generated, GenerationResult):
            add_finding(question_id, "generate_return_type", ERROR_STR,
                "generate() must return a GenerationResult.",
            )
            result.questions.append(
                PreflightQuestionResult(question_id, "invalid")
            )
            continue

        if not generated.answer or not generated.answer.strip():
            add_finding(question_id, "empty_answer", ERROR_STR,
                "generate() returned an empty or whitespace-only answer.",
            )

        result.questions.append(
            PreflightQuestionResult(question_id, "checked")
        )

    return result