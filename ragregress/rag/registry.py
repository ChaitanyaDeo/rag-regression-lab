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
"""

from __future__ import annotations
 
from dataclasses import dataclass
from typing import Callable, Dict, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator
 
from ragregress.core.errors import PipelineContractError, UnknownPipelineError
from ragregress.rag.interface import RAGSystem

class PipelineParams(BaseModel):
    """Base for every pipeline's params model.
 
    extra="forbid": a typo like `chunck_size` is rejected.
    frozen=True: keeps config_hash stable for the whole run.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

MAX_CHUNK_SIZE = 4000


class BuiltinRAGParams(PipelineParams):
    """Knobs owned by BuiltinRAGSystem. All required"""

    embedding_model: str = Field(min_length=1)
    chunk_size: int = Field(ge=1, le=MAX_CHUNK_SIZE)
    overlap: int = Field(ge=0)
    generation_model: str = Field(min_length=1)
 
    @model_validator(mode="after")
    def _overlap_less_than_chunk_size(self) -> "BuiltinRAGParams":
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


def _build_builtin_rag(params: BuiltinRAGParams) -> RAGSystem:
    # Lazy imports: only paid when this pipeline is actually built.
    from ragregress.rag.builtin_system import BuiltinRAGSystem
    from ragregress.rag.chunker import ChunkerConfig
    from ragregress.rag.generator import GeneratorConfig
 
    return BuiltinRAGSystem(
        ChunkerConfig(chunk_size=params.chunk_size, overlap=params.overlap),
        generator_config=GeneratorConfig(model_name=params.generation_model),
        embedding_model=params.embedding_model,
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