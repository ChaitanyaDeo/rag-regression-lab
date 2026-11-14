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
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, ClassVar, Dict, Literal, Optional, Type, Union

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
from ragregress.rag.prompts import (
    DEFAULT_PROMPT,
    PROMPT_NAME_PATTERN,
    load_validated,
    prompt_sha256,
)

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


MAX_WINDOW = 20
DEFAULT_CHUNKING = "fixed"


class FixedChunking(PipelineParams):
    """Fixed windows of chunk_size characters. Same as leaving `chunking` out."""

    strategy: Literal["fixed"]
    uses_overlap: ClassVar[bool] = True

class RecursiveChunking(PipelineParams):
    """Breaks at a blank line, a line end or a space, never mid-word."""

    strategy: Literal["recursive"]
    uses_overlap: ClassVar[bool] = True

class SentenceChunking(PipelineParams):
    """Whole sentences packed up to chunk_size; `overlap` = characters of trailing sentences repeated."""

    strategy: Literal["sentence"]
    uses_overlap: ClassVar[bool] = True


class SlidingWindowChunking(PipelineParams):
    """`window` sentences per chunk, moving on `stride` sentences; chunk_size is the length limit.

    Overlap comes from window - stride shared sentences, so `overlap` must be 0.
    """

    strategy: Literal["sliding_window"]
    window: int = Field(ge=1, le=MAX_WINDOW)
    stride: int = Field(ge=1, le=MAX_WINDOW)
    uses_overlap: ClassVar[bool] = False

    @model_validator(mode="after")
    def _stride_not_larger_than_window(self) -> "SlidingWindowChunking":
        if self.stride > self.window:
            raise ValueError(f"stride ({self.stride}) must not be larger than window ({self.window})")
        return self

# Must stay in step with ragregress.rag.chunker.STRATEGIES (tests/unit/test_chunking_config.py checks it).
ChunkingParams = Annotated[
    Union[FixedChunking, RecursiveChunking, SentenceChunking, SlidingWindowChunking],
    Field(discriminator="strategy"),
]


# Optional sub-models omitted from the serialized params when None (H3).
OMIT_WHEN_NONE = ("reranker", "chunking",)


class BuiltinRAGParams(PipelineParams):
    """Knobs owned by BuiltinRAGSystem. All required except seed, reranker, chunking and prompt"""

    embedding_model: str = Field(min_length=1)
    chunk_size: int = Field(ge=1, le=MAX_CHUNK_SIZE)
    overlap: int = Field(ge=0)
    generation_model: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0)
    reranker: Optional[RerankerParams] = None
    chunking: Optional[ChunkingParams] = None      # absent = the fixed strategy
    prompt:  str = Field(default=DEFAULT_PROMPT, pattern=PROMPT_NAME_PATTERN)   # a file in ragregress/rag/prompts/
    prompt_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")   # filled in; see _resolve_prompt


    @model_validator(mode="before")
    @classmethod
    def _resolve_prompt(cls, data:Any):
        """A non-default prompt records the sha256 of its text, so editing the file changes config_hash.

        Read once, when the config is loaded. A stored config already carries the sha and is not touched
        (comparing old runs never needs the file). Unknown name, missing file or bad placeholders: a config error.
        """

        if not isinstance(data, dict):
            return data
        name = data.get("prompt", DEFAULT_PROMPT)

        if not isinstance(name, str) or name == DEFAULT_PROMPT or re.fullmatch(PROMPT_NAME_PATTERN, name) is None:
            return data                                    # the field's own check reports a bad name

        if data.get("prompt_sha256"):
            return data 

        # a stored config: the file is not needed
        text = load_validated(name)                        # PromptError is a ValueError: reported as a config error
        return {**data, "prompt_sha256": prompt_sha256(text)}

    

    @model_validator(mode="after")
    def _chunking_is_compatible_with_overlap(self) -> "BuiltinRAGParams":
        if self.chunking is not None and not self.chunking.uses_overlap and self.overlap != 0:
            raise ValueError(
                f"chunking strategy {self.chunking.strategy!r} does not use overlap; set overlap to 0"
            )
        return self

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

            if data.get("chunking") == {"strategy": DEFAULT_CHUNKING}:      # explicit fixed == absent
                del data["chunking"]

            if data.get("prompt") == DEFAULT_PROMPT:
                data.pop("prompt", None)
                data.pop("prompt_sha256", None)
        return data


def chunker_kwargs(params: BuiltinRAGParams) -> Dict[str, Any]:
    """Arguments for ChunkerConfig. strategy / options are passed ONLY for a non-default strategy,
    so with `chunking` off the call is the one made before chunking strategies existed."""
    
    kwargs: Dict[str, Any] = dict(chunk_size=params.chunk_size, overlap=params.overlap)
    chunking = params.chunking
    if chunking is not None and chunking.strategy != DEFAULT_CHUNKING:
        kwargs["strategy"] = chunking.strategy
        options = chunking.model_dump(exclude={"strategy"})
        if options:
            kwargs["options"] = options
    return kwargs


def _build_builtin_rag(params: BuiltinRAGParams) -> RAGSystem:
    # Lazy imports: only paid when this pipeline is actually built.
    from ragregress.rag.builtin_system import BuiltinRAGSystem
    from ragregress.rag.chunker import ChunkerConfig
    from ragregress.rag.generator import GeneratorConfig

    # `prompt` is passed ONLY when it is not the default, like the reranker kwargs below. The file must still
    # be the one the config was made with (its sha256 is part of config_hash).
    generator_kwargs = dict(model_name=params.generation_model, seed=params.seed)
    if params.prompt != DEFAULT_PROMPT:
        if prompt_sha256(load_validated(params.prompt)) != params.prompt_sha256:
            raise PipelineContractError(
                f"prompt {params.prompt!r} changed since this config was created (its sha256 no longer matches "
                f"the config's); reload the config from the YAML"
            )
        generator_kwargs["prompt"] = params.prompt

    kwargs = dict(
        generator_config=GeneratorConfig(**generator_kwargs),
        embedding_model=params.embedding_model,
    )
    # Reranker kwargs are passed ONLY when configured: with the feature off the
    # constructor call is identical to the pre-reranker one (F-D12-23).
    if params.reranker is not None:
        from ragregress.rag.reranker import CrossEncoderReranker

        kwargs["reranker"] = CrossEncoderReranker(params.reranker.model)
        kwargs["rerank_candidates"] = params.reranker.candidates

    return BuiltinRAGSystem(
        ChunkerConfig(**chunker_kwargs(params)),
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