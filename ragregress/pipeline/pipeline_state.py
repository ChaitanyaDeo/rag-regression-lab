"""
LangGraph sate definition

2 sates:
- Run State: One instance per graph execution. Holds run-wide
  config/status and the reduced list of per-query results.

- Query State: One instance per QA pair, dispatched via LangGraph's Send()
  API after Ingest. Carries everything a single query's Retrieve -> Generate
  -> Judge branch needs, independent of every other branch.


"""

from __future__ import annotations
import operator
from typing import Annotated, List, Dict, Optional, TypedDict, Any

from ragregress.benchmark.schemas import QAPair
from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.evaluation.metrics.base import MetricResult
from ragregress.pipeline.pipeline_constants import QueryStatus, RunStatus


# Run level state

class RunState(TypedDict, total=False):
    """Shared state for one full pipeline run.
    per_query_results uses the `operator.add` reducer: each terminal branch
    
    returns a ONE-ITEM list here, and LangGraph concatenates them as branches finish.

    each item carries its own qa_id, and
    Aggregate must key/sort by qa_id rather than relying on list order.
    """

    run_id: str
    config: Dict[str, str]  # TODO replace with RunConfig

    corpus_loaded: bool
    index_built: bool

    qa_pairs: List[QAPair]

    # Ingest sets this to RunStatus.HEALTHY/ RunStatus.UNHEALTHY
    run_status: RunStatus
    run_status_detail: Optional[str]

    # Reduced fan-in from every query branch
    per_query_results: Annotated[List[Dict[str, Any]], operator.add]

    aggregated_metrics: Dict[str, Any]


# Per branch state

class QueryState(TypedDict, total=False):
    """State for a single QA pair's Retrieve -> Generate -> Judge branch.
    """

    qa_id: str
    question: str
    gold_ids: List[str]
    gold_answers: List[str]

    chunks: Optional[List[Chunk]]
    generation: Optional[GenerationResult]

    # Keyed by metric name (Metric.name),
    metric_results: Dict[str, MetricResult]

    # Per-stage retry counters, read by that stage's own conditional edge.
    # e.g. {"retrieve": 0, "generate": 0} -- Judge has no retry
    retry_count: Dict[str, int]

    last_error : Optional[str]

    # Derived at Aggregate
    failure_mode: Optional[str]

    # Per-stage wall-clock time in ms,
    latency_ms: Dict[str, float]

    # pending -> retrieving -> generating -> judging -> judged -> failed (terminal)
    query_status: QueryStatus

    per_query_results: Annotated[List[Dict[str, Any]], operator.add]

    category: Optional[str]


def new_query_state(qa: QAPair) -> QueryState:
    """Build the initial QueryState for one QA pair, for Send() dispatch.

    Returns:
    - QueryState
    """

    if isinstance(qa, dict):
        qa = QAPair(**qa)

    return QueryState(
        qa_id=qa.qa_id,
        question=qa.question,
        gold_ids=qa.gold_passage_ids,
        gold_answers=qa.answer,
        chunks=None,
        generation=None,
        metric_results={},
        retry_count={"retrieve": 0, "generate": 0},
        failure_mode=None,
        latency_ms={},
        query_status=QueryStatus.PENDING,
        per_query_results=[],
        last_error = None,
        category=getattr(qa, "category", None)
    )