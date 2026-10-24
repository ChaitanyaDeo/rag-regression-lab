"""
ragregress/core/config.py

RunConfig: the validated, immutable description of one evaluation run

Loaded from YAML by load_run_config().

Shape:
    pipeline:            <- owned by the pipeline under test
      id: builtin_rag
      params: {...}      <- validated against the registry's ParamsModel
    top_k: 5             <- harness-owned fields, top level
    judge_model: ...
    eval_split: val
    is_dev: false
    max_workers: 1       <- operational, defaulted, not hashed
    db_path: ragregress.db

Decisions:
  - Every experiment-defining field is required, except judge_seed and
    pipeline.params.seed (default 42, hashed); max_workers and db_path have defaults (not hashed)
  - extra="forbid" at every level
  - max_workers capped at 8 as a guardrail
  - frozen=True at every level.
  - is_dev must equal (eval_split == "dev"), both directions.
  - `top_k` is the config-facing name only: it maps once to
    Pipeline(k=...); everything below the config boundary keeps `k`.
  - config_hash() excludes db_path and max_worker: so reruns of the same experiment share a hash.
"""

from __future__ import annotations
 
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ragregress.core.errors import ConfigValidationError
from ragregress.rag.registry import PIPELINE_REGISTRY, PipelineParams

MAX_TOP_K = 50
MAX_WORKERS_CAP = 8
DATA_DIR = Path("data")
HASH_EXCLUDE = {"db_path", "max_workers"}


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)

    params: SerializeAsAny[PipelineParams]

    @field_validator("params", mode="before")
    @classmethod
    def _validate_params_against_registry(cls, value: Any, info: ValidationInfo) -> Any:
        pipeline_id = info.data.get("id")
        if pipeline_id is None:
            # `id` itself failed validation; its own error is already reported.
            return value
        entry = PIPELINE_REGISTRY.get(pipeline_id)
        if entry is None:
            raise ValueError(
                f"unknown pipeline id {pipeline_id!r}; registered: {sorted(PIPELINE_REGISTRY)}"
            )
        return entry.params_model.model_validate(value)



class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
 
    pipeline: PipelineSpec
    top_k: int = Field(ge=1, le=MAX_TOP_K)
    judge_model: str = Field(min_length=1)
    judge_seed: int = Field(default=42, ge=0)
    eval_split: Literal["dev", "val", "test"]
    is_dev: bool
 
    max_workers: int = Field(default=1, ge=1, le=MAX_WORKERS_CAP)
    db_path: Path = Path("ragregress.db")

    @model_validator(mode="after")
    def _is_dev_matches_eval_split(self) -> "RunConfig":
        expected = self.eval_split == "dev"
        if self.is_dev != expected:
            raise ValueError(
                f"is_dev={self.is_dev} does not match eval_split={self.eval_split!r} "
                f"(expected is_dev={expected})"
            )
        return self

    def split_path(self, data_dir: Path = DATA_DIR) -> Path:
        return data_dir / "splits" / f"{self.eval_split}.json"
 
    def hash_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude=HASH_EXCLUDE)
 
    def config_hash(self) -> str:
        canonical = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _format_loc(loc: tuple) -> str:
    return ".".join(str(part) for part in loc) or "<root>"


def load_run_config(path: Union[str, Path]) -> RunConfig:
    """Load and validate a RunConfig from YAML.
 
    Every failure (missing file, bad YAML, validation) raises
    ConfigValidationError -- never a raw ValidationError/YAMLError.
    """
    path = Path(path)
    source = str(path)
 
    if not path.is_file():
        raise ConfigValidationError("config file not found", source=source)
 
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"invalid YAML: {exc}", source=source) from exc
 
    if not isinstance(raw, dict):
        raise ConfigValidationError("top level of the config must be a mapping", source=source)
 
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        errors = [(_format_loc(err["loc"]), err["msg"]) for err in exc.errors()]
        raise ConfigValidationError(
            f"{len(errors)} validation error(s)", errors=errors, source=source
        ) from exc
