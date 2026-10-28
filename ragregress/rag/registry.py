"""
ragregress/rag/registry.py

pipeline_id -> RAGSystem resolution

Each registry entry is (ParamsModel, factory):
  - ParamsModel: the Pydantic model for that pipeline's own knobs, i.e.
    the `pipeline.params` block of a run config. RunConfig validates
    `params` against it by looking up `pipeline.id` here, so this
    registry is the single source of truth for which pipelines exist
    and what each one accepts.

  - factory: (params) -> RAGSystem. Imports its pipeline lazily, so
    importing this module (which config.py does on every load) never
    pulls in heavy deps like sentence-transformers

Protocol enforcement - in build_rag_system() right after construction

Adding a pipeline = one ParamsModel + one factory + one registry line.

Optional features: an optional knob is ONE optional sub-model
that defaults to None (= feature off). When None it is omitted from the
serialized params, so it never enters config_hash or the stored run config.
Existing configs therefore keep their hashes when a feature is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from ragregress.core.errors import PipelineContractError, UnknownPipelineError
from ragregress.rag.interface import RAGSystem

class PipelineParams(BaseModel):
    """Base for every pipeline's params model.

    extra="forbid": a typo like `chunck_size` is rejected.
    frozen=True: keeps config_hash stable for the whole run.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

MAX_CHUNK_SIZE = 4000
MAX_RERANK_CANDIDATES = 50   # = config.MAX_TOP_K (config.py imports this module, so no import back)


class RerankerParams(PipelineParams):
    """Cross-encoder reranking of the vector-search candidates (BuiltinRAGSystem only).

    Present in the YAML = on; absent (or null) = off (D12-Q10).
    `candidates` = how many chunks vector search returns before reranking
    down to top_k; RunConfig rejects candidates < top_k (D12-Q8).
    """

    model: str = Field(min_length=1)
    candidates: int = Field(ge=1, le=MAX_RERANK_CANDIDATES)


# Optional sub-models omitted from the serialized params when None (H3).
OMIT_WHEN_NONE = ("reranker",)


class BuiltinRAGParams(PipelineParams):
    """Knobs owned by BuiltinRAGSystem. All required except seed and reranker"""

    embedding_model: str = Field(min_length=1)
    chunk_size: int = Field(ge=1, le=MAX_CHUNK_SIZE)
    overlap: int = Field(ge=0)
    generation_model: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0)
    reranker: Optional[RerankerParams] = None

    @model_validator(mode="after")
    def _overlap_less_than_chunk_size(self) -> "BuiltinRAGParams":
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_disabled_features(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        # Every model_dump (config_hash payload, stored run config, dry-run print)
        # goes through here, so "feature off" has exactly one serialized form: absent.
        data = handler(self)
        for name in OMIT_WHEN_NONE:
            if data.get(name, "<set>") is None:
                del data[name]
        return data


def _build_builtin_rag(params: BuiltinRAGParams) -> RAGSystem:
    # Lazy imports: only paid when this pipeline is actually built.
    from ragregress.rag.builtin_system import BuiltinRAGSystem
    from ragregress.rag.chunker import ChunkerConfig
    from ragregress.rag.generator import GeneratorConfig

    kwargs = dict(
        generator_config=GeneratorConfig(model_name=params.generation_model, seed=params.seed),
        embedding_model=params.embedding_model,
    )
    # Reranker kwargs are passed ONLY when configured: with the feature off the
    # constructor call is identical to the pre-reranker one (F-D12-23).
    if params.reranker is not None:
        from ragregress.rag.reranker import CrossEncoderReranker

        kwargs["reranker"] = CrossEncoderReranker(params.reranker.model)
        kwargs["rerank_candidates"] = params.reranker.candidates

    return BuiltinRAGSystem(
        ChunkerConfig(chunk_size=params.chunk_size, overlap=params.overlap),
        **kwargs,
    )


@dataclass(frozen=True)
class PipelineEntry:
    params_model: Type[PipelineParams]
    factory: Callable[[PipelineParams], object]


PIPELINE_REGISTRY: Dict[str, PipelineEntry] = {
    "builtin_rag": PipelineEntry(BuiltinRAGParams, _build_builtin_rag),
}

def get_entry(pipeline_id: str) -> PipelineEntry:
    try:
        return PIPELINE_REGISTRY[pipeline_id]
    except KeyError:
        raise UnknownPipelineError(
            f"unknown pipeline id {pipeline_id!r}; registered: {sorted(PIPELINE_REGISTRY)}"
        ) from None


def build_rag_system(pipeline_id: str, params: PipelineParams) -> RAGSystem:
    """Construct the RAGSystem for `pipeline_id` and verify the protocol."""
    entry = get_entry(pipeline_id)

    if not isinstance(params, entry.params_model):
        raise PipelineContractError(
            f"pipeline {pipeline_id!r} expects {entry.params_model.__name__}, "
            f"got {type(params).__name__}"
        )

    system = entry.factory(params)

    if not isinstance(system, RAGSystem):
        raise PipelineContractError(
            f"pipeline {pipeline_id!r} built {type(system).__name__}, which does not "
            f"satisfy the RAGSystem protocol (needs retrieve() and generate())"
        )
    return system