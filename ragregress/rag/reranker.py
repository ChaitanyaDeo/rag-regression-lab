"""
ragregress/rag/reranker.py

Cross-encoder reranking for BuiltinRAGSystem 

Internal to BuiltinRAGSystem only - not part of public RAGSystem contract.

BuiltinRAGSystem.retrieve() asks vector search for `candidates` chunks, then
CrossEncoderReranker.rerank() scores each (query, chunk text) pair with a
cross-encoder and keeps the best k. Only the ORDER and the SELECTION of chunks
change: chunks are returned as the same objects, unmodified.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from ragregress.rag.interface import Chunk

@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: List[Chunk], k: int) -> List[Chunk]:
        """Return the top k of chunks for query, best first"""
        ...


class CrossEncoderReranker:
    """sentence-transformer CrossEncoder wrapper

    `model` is for tests: pass any object with predict(list_of_(query, text)_pairs)
    and the real model is never loaded. Otherwise `model_name` is loaded lazily
    here, so importing this module never pulls in sentence-transformers.

    Scores are the model's raw outputs (logits); only their order matters.
    Ties keep the incoming (vector-search) order; a NaN score ranks last.
    """

    def __init__(self, model_name:str, device: Optional[str] = None, model = None):
        self.model_name = model_name
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device=device)

        self.model = model

    def rerank(self, query: str, chunks: List[Chunk], k: int) -> List[Chunk]:
        if k<1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not chunks:
            return []

        raw = self.model.predict([(query, c.text) for c in chunks])
        scores = [float(s) for s in raw]

        if len(scores)!=len(chunks):
            raise RuntimeError(f"cross-encoder returned {len(scores)} scores for {len(chunks)} chunks")
        scores = [float("-inf") if s != s else s for s in scores]

        order = sorted(range(len(chunks)), key = lambda i: (-scores[i], i))
        return [chunks[i] for i in order[:k]]
