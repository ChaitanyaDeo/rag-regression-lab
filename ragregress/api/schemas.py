"""
ragregress/api/schemas.py

Request and response models for the API. They are separate from the engine's classes on purpose:
the JSON contract does not change when internals do.

    ID_PATTERN / NAME_PATTERN   allowed characters for run ids and config names (no "/" or "..").
    MAX_CANDIDATES              limit for one /compare request (10).
    clean_number(x)             None for NaN and +/-inf (not valid JSON); other values unchanged.

    Health                      GET /health
    RunRequest, RunAccepted     POST /runs
    RunSummary, RunDetail       GET /runs, GET /runs/{id}; JobView is the live job state.
                                RunDetail.is_dev and .k are None while the run's row is not written yet.
    CompareRequest              POST /compare: 1..10 unique candidate ids; unknown keys rejected.
    CompareResponse             one CompareEntry per candidate: ok, error, or a ComparisonOut
                                (verdict, metrics, validity, config changes, RunRef, ...).
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict, field_validator
import math
import re

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
MAX_CANDIDATES = 10
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"

class Health(BaseModel):
    status: str
    ollama_reachable: bool
    judge_reachable: bool


class JobView(BaseModel):
    state: str
    started_at: str
    finished_at: Optional[str] = None
    rows_done: int
    rows_total: Optional[int] = None
    error: Optional[str] = None

class RunSummary(BaseModel):
    run_id: str
    created_at: str
    is_dev: bool
    k: int
    run_status: Optional[str] = None
    eval_split: Optional[str] = None
    pipeline_id: Optional[str] = None
    config_hash: Optional[str] = None

class RunDetail(RunSummary):
    is_dev: Optional[bool] = None
    k: Optional[int] = None
    run_status_detail: Optional[str] = None
    config: Dict[str, Any]
    metrics: Dict[str, Optional[float]]
    job: Optional[JobView] = None



def clean_number(value: Optional[float]) -> Optional[float]:

    if value is None:
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value

class CompareRequest(BaseModel):
    baseline: str = Field(pattern=ID_PATTERN)
    candidates : List[str] = Field(min_length=1, max_length=MAX_CANDIDATES)

    model_config = ConfigDict(extra="forbid")

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, candidates: List[str]) -> List[str]:
        for c in candidates:
            if not re.fullmatch(ID_PATTERN, c):
                raise ValueError(f"Invalid candidate_id: {c}")

        if len(candidates) != len(set(candidates)):
            raise ValueError("Duplicate candidate IDs are not allowed")

        return candidates


# Response Models

class RunRef(BaseModel):
    run_id : str
    config_hash : Optional[str] = None
    eval_split: Optional[str] = None
    top_k: int
    legacy: bool

class MetricOut(BaseModel):

    name: str
    status: str
    before: Optional[float] = None
    after: Optional[float] = None
    delta: Optional[float] = None
    threshold: Optional[float] = None
    n_common: int
    better: int
    worse: int
    tie: int
    higher_is_better: Optional[bool] = None
    note: str

class ConfigChangeOut(BaseModel):
    key: str
    before: Any
    after: Any
    before_missing: bool
    after_missing: bool

class ValidityOut(BaseModel):
    measured: int
    attempted: int
    ratio: float

class ComparisonOut(BaseModel):
    verdict: str
    exit_code: int
    reasons: List[str]
    warnings: List[str]
    gated: List[str]
    baseline: RunRef
    candidate: RunRef
    validity_baseline: ValidityOut
    validity_candidate: ValidityOut
    config_changes: List[ConfigChangeOut]
    metrics: List[MetricOut]

class CompareEntry(BaseModel):
    candidate: str
    ok: bool
    error: Optional[str] = None
    comparison: Optional[ComparisonOut] = None

class CompareResponse(BaseModel):
    baseline: str
    results: List[CompareEntry]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str = Field(pattern=NAME_PATTERN)


class RunAccepted(BaseModel):
    run_id: str
    state: str
    status_url: str