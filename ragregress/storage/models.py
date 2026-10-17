"""
JSON-friendly dataclasses representing the storage layer's own rows.

These are distinct on purpose from:
- RunState / QueryState (pipeline_state.py) -- TypedDicts used as LangGraph
  channels during graph execution, not meant to be persisted as-is.
- MetricResult (evaluation/metrics/base.py) -- the in-memory result of one
  metric's .score() call, used during scoring/judging


"""

from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Run:
    run_id: str
    created_at: str
    is_dev: bool
    k: int
    max_workers: int
    config: dict[str, Any]
    run_status: str | None = None
    run_status_detail: str | None = None
    config_hash: str | None = None


@dataclass
class RunMetric:
    run_id: str
    metric_name: str
    value: float | None


@dataclass
class QueryResult:
    run_id: str
    qa_id: str
    question: str | None = None
    query_status: str | None = None
    failure_mode: str | None = None
    last_error: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    gold_ids: list[str] = field(default_factory=list)
    generation_answer: str | None = None
    metric_results: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    category: str | None = None