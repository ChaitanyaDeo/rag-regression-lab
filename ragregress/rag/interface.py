"""

RAGSystem contract for the regression-testing harness.

THIS is the module users implement against to plug in their own pipeline.

A user writes their own class satisfying RAGSystem (retrieve/generate)
and returning Chunk objects with a valid source_passage_id — that is the 
entire integration surface.

They never touch chunker.py, embedder.py,
retriever.py, or builtin_system.py; those are internal to this lab's own
reference pipeline, BuiltinRAGSystem, not part of the public contract.

Every RAG pipeline calls a RAGSystem's retrieve()/generate() —
never the underlying chunker/embedder/retriever modules directly.

"""

from typing import Any, Dict, List, Protocol, runtime_checkable

from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """A retrieved unit of text.

    source_passage_id MUST resolve to a real id in data/passages.json.
    
    It is never a pipeline-internal chunk id — that's chunk_id, which is
    only used to identify the chunk within the vector store (e.g. for
    upsert idempotency), not for scoring against the eval corpus.
    """

    chunk_id: str
    source_passage_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Output of a RAGSystem's generate() call.

    raw_output: exactly what the LLM generates

    answer: citation-stripped text.

    citations: parsed passage_ids, in order of first appearance
    """

    raw_output: str      # exactly what the LLM returned
    answer: str         # citation-stripped text — this is what answer_f1() scores against
    citations: List[str] = Field(default_factory=list)       # List of citation 



@runtime_checkable
class RAGSystem(Protocol):

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        """Return the top-k retrieved chunks for the given query"""
        ...

    def generate(self, query: str, contexts: List[Chunk]) -> GenerationResult:
        """Return the generated answer for the given query and retrieved contexts"""
        ...


