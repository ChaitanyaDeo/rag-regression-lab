"""
The pipeline of whole evaluation

LangGraph state graph orchestration for run.

Ingest -> Retrive -> Generate -> judge -> Aggregate -> persists -> Compare

Logging: every stage emits one structured record through the `ragregress.pipeline.pipeline`
logger (schema in ragregress/core/logging.py: run_id, stage, query_id, attempt, status, latency_ms,
tokens_in, tokens_out, error). Handlers are configured by the entry point (CLI / API), never here.
Nothing in this module prints.
"""



import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, Callable
from datetime import datetime, timezone
import threading

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ragregress.benchmark.schemas import QAPair
from ragregress.rag.interface import RAGSystem
from ragregress.evaluation.metrics.base import Metric
from ragregress.evaluation.metrics import RecallAtK, TokenPRF1, Faithfulness
from ragregress.pipeline.pipeline_state import RunState, QueryState, new_query_state

from ragregress.pipeline.pipeline_constants import (
    Node,
    Route,
    QueryStatus,
    RunStatus,
    FailureMode,
    RUN_STATUS_DETAILS, RUN_STATUS, FAILURE_MODE, PER_QUERY_RESULT, QUERY_STATUS,
    QA_ID, QUESTION, GOLD_IDS, GENERATION, METRIC_RESULTS, LAST_ERROR, LATENCY_MS,
    CATEGORY, CHUNKS
)
if TYPE_CHECKING:
    from ragregress.core.config import RunConfig     # Type-only: keeps the pipeline free of a runtime dependency on config.py
from ragregress.storage.database import DB_PATH_DEFAULT, init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

logger = logging.getLogger(__name__)

GENERATION_FIELDS = {"answer", "context"}
MAX_RETRIES = 3

FAITHFULNESS_LOW_THRESHOLD = 0.5
TOKEN_PRF1_LOW_THRESHOLD = 0.3

LATENCY_STAGES = (Node.RETRIEVE, Node.GENERATE, Node.JUDGE)

# Values of the log `status` field (stage outcome; distinct from QueryStatus / RunStatus)
LOG_OK = "ok"
LOG_SKIPPED = "skipped"         # ingest: the RAG system manages its own index
LOG_FAILED = "failed"
LOG_PARTIAL = "partial"            # judge: some metrics scored, some crashed
LOG_METRIC_FAILED = "metric_failed"
LOG_STARTED = "started"
LOG_CRASHED = "crashed"
RUN_STAGE = "run"                  # stage name for run start / finish records (not a graph node)


def _percentile(values, pct):
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return float(ordered[idx])

# Helpers
def _now_ms() -> float:
    return time.perf_counter() * 1000


def _snapshot(qs: QueryState, update: Dict[str, Any]) -> Dict[str, Any]:
    """Build a terminal branch's final record for the per_query_results
    reducer: the branch's current state with `update` applied on top,
    minus the reducer field itself (so a row never nests an empty copy of
    per_query_results inside itself).
    """
    merged = {**qs, **update}
    merged.pop("per_query_results", None)
    return merged


class Pipeline:
    _run_config: Optional["RunConfig"] = None
    _run_id: Optional[str] = None       # set by run(); carried on every log record
    on_row_done: Optional[Callable[[int, int, float], None]] = None

    def __init__(
        self,
        rag_system: RAGSystem,
        metrics: List[Metric],
        k: int = 5,
        max_workers: int = 1,  # default sequential (7B production run)
        db_path=DB_PATH_DEFAULT,
    ):
        self.rag_system = rag_system
        self.metrics = metrics
        self.k = k
        self.max_workers = max_workers

        # List of variables needed for evaluations
        needed = set().union(*(m.required_inputs for m in metrics)) if metrics else set()

        self.needs_generation = bool(needed & GENERATION_FIELDS)

        self.graph = self._build_graph().compile()

        self.db_path = db_path
        init_db(db_path)
        self._repo = Repository(db_path)

    @classmethod
    def from_config(cls, run_config:"RunConfig", rag_system: RAGSystem,metrics: List[Metric],) -> "Pipeline":
        """Build a Pipeline whose k, max_workers, db_path, is_dev, stored config
        and config_hash all come from one validated RunConfig

        rag_system and metrics are still supplied by the caller
        """

        pipeline = cls(
            rag_system=rag_system,
            metrics=metrics,
            k=run_config.top_k,  # config-facing name -> internal `k` (DEC-D7-06)
            max_workers=run_config.max_workers,
            db_path=run_config.db_path,
        )

        pipeline._run_config = run_config

        return pipeline

    # Structured logging

    def _log(self, level: int, event: str, *, stage, status: str, query_id: Optional[str] = None,
             attempt: Optional[int] = None, latency_ms: Optional[float] = None,
             error: Optional[str] = None, exc_info: bool = False) -> None:
        """One structured record (schema: ragregress/core/logging.py LOG_FIELDS).

        tokens_in / tokens_out stay None: GenerationResult carries no token counts yet (B1-Q5).
        """
        logger.log(level, event, exc_info=exc_info, extra={
            "run_id": self._run_id,
            "stage": str(stage),
            "query_id": query_id,
            "attempt": attempt,
            "status": status,
            "latency_ms": None if latency_ms is None else round(latency_ms, 1),
            "tokens_in": None,
            "tokens_out": None,
            "error": error,
        })

    # Graph construction

    def _build_query_subgraph(self) -> StateGraph:
        """Per-batch subgraph:
            etrieve -> Generate -> Judge

        Operates entirely onQueryState

        Compiled once and embedded as a single node in the parent graph
        """

        sg = StateGraph(QueryState)

        sg.add_node(Node.RETRIEVE, self._retrieve)
        sg.add_node(Node.GENERATE, self._generate)

        sg.add_node(Node.JUDGE, self._with_progress(self._judge))       # progress bar wrapper for terminal node
        sg.add_node(Node.FAILED_QUERY, self._with_progress(self._failed_query))

        sg.add_edge(START, Node.RETRIEVE)

        # Rrtrieve Success-> Generate or judge; failure -> retry or Failure_i
        sg.add_conditional_edges(
            Node.RETRIEVE, self._route_after_retrieve,
            {
                Node.GENERATE: Node.GENERATE,
                Node.JUDGE: Node.JUDGE,
                Route.RETRY: Node.RETRIEVE,
                Node.FAILED_QUERY: Node.FAILED_QUERY
            }
        )

        # Generate success -> Judge; failure -> retry or Failure_i
        sg.add_conditional_edges(
            Node.GENERATE, self._route_after_generate,
            {
                Node.JUDGE: Node.JUDGE,
                Route.RETRY: Node.GENERATE,
                Node.FAILED_QUERY: Node.FAILED_QUERY
            }
        )

        sg.add_edge(Node.JUDGE, END)
        sg.add_edge(Node.FAILED_QUERY, END)

        return sg


    def _build_graph(self) -> StateGraph:
        """Parent Graph:
            Ingest -> Send() fan out into compiled subgraph

            One Invocation per:
            QA Pair -> Aggregrate -> Persist -> Compare.
        """

        g = StateGraph(RunState)

        g.add_node(Node.INGEST, self._ingest)

        # The compiled subgraph IS this node's action -- Send(Node.RETRIEVE, ...)
        # dispatches a fresh, isolated invocation of it per QA pair

        g.add_node(Node.PROCESS_QUERY, self._build_query_subgraph().compile())

        g.add_node(Node.FAILED_RUN, self._failed_run)
        g.add_node(Node.AGGREGATE, self._aggregate)
        g.add_node(Node.PERSIST, self._persist)
        g.add_node(Node.COMPARE, self._compare)

        g.add_edge(START, Node.INGEST)

        # Ingest always runs to completion; THIS edge decides
        # fan-out vs. fail-fast, not the Ingest node itself.

        g.add_conditional_edges(
            Node.INGEST, self._route_after_ingest,
            {
                Route.FAN_OUT: Node.PROCESS_QUERY,
                Node.FAILED_RUN: Node.FAILED_RUN
            }
        )

        # Every sub-branch reaches here after its own END
        g.add_edge(Node.PROCESS_QUERY, Node.AGGREGATE)

        g.add_edge(Node.FAILED_RUN, Node.PERSIST)

        g.add_edge(Node.AGGREGATE, Node.PERSIST)
        g.add_edge(Node.PERSIST, Node.COMPARE)
        g.add_edge(Node.COMPARE, END)

        return g

    # Progress bar wrappers
    def _with_progress(self, node_fn):
        """Wrap a terminal per-row node: after it runs, count the row as finished."""
        def wrapper(qs: QueryState):
            update = node_fn(qs)
            self._row_finished()
            return update
        wrapper.__name__ = node_fn.__name__
        return wrapper

    def _row_finished(self):
        cb = self.on_row_done
        if cb is None:
            return
        with self._progress_lock:
            self._rows_done += 1
            done, total = self._rows_done, self._rows_total

        try:
            cb(done, total, time.monotonic() - self._t_start)
        except Exception:
            pass

    def _start_progress(self, total: int):
        self._progress_lock = threading.Lock()
        self._rows_done = 0
        self._rows_total = total
        self._t_start = time.monotonic()
        if self.on_row_done is not None:
            try:
                self.on_row_done(0, total, 0.0)
            except Exception:
                pass


    # Ingest Run Level
    def _ingest(self, state: RunState) -> Dict[str, Any]:
        """Load corpus/index, then health check the RAG itself
        """
        t0 = _now_ms()
        updates: Dict[str, Any] = {}

        if hasattr(self.rag_system, "ingest"):
            # Caller is expected to have already placed passages into
            # state["config"]["passages"] if BuiltinRAGSystem-style
            # ingestion is needed; this pipeline does not assume any one
            # RAGSystem's ingest() signature beyond "callable with the
            # passages it was given".
            passages = state.get("config", {}).get("passages")
            if passages:
                self.rag_system.ingest(passages)
            updates["corpus_loaded"] = True
            updates["index_built"] = True

        else:
            # trust it manages its own indexing externally.
            # Not silent: logged for run visibility
            self._log(logging.INFO,
                      f"ingest: no ingest() on {type(self.rag_system).__name__}, assuming pre-built index",
                      stage=Node.INGEST, status=LOG_SKIPPED)

            updates["corpus_loaded"] = True
            updates["index_built"] = True

        try:
            self.rag_system.retrieve("health check probe", k=1)
            updates[RUN_STATUS] = RunStatus.HEALTHY
            updates[RUN_STATUS_DETAILS] = None
        except Exception as e:
            updates[RUN_STATUS] = RunStatus.UNHEALTHY
            updates[RUN_STATUS_DETAILS] = repr(e)

        if updates[RUN_STATUS] == RunStatus.HEALTHY:
            self._log(logging.INFO, "ingest: health probe ok", stage=Node.INGEST,
                      status=str(RunStatus.HEALTHY), latency_ms=_now_ms() - t0)
        else:
            self._log(logging.ERROR, "ingest: health probe failed, run is unhealthy", stage=Node.INGEST,
                      status=str(RunStatus.UNHEALTHY), latency_ms=_now_ms() - t0,
                      error=updates[RUN_STATUS_DETAILS])

        updates["qa_pairs"] = state["qa_pairs"]

        return updates

    def _route_after_ingest(self, state: RunState)-> Union[str, List[Send]]:
        if state.get(RUN_STATUS) != RunStatus.HEALTHY:
            return Node.FAILED_RUN

        qa_pairs = state["qa_pairs"]
        return [Send(Node.PROCESS_QUERY, new_query_state(qa)) for qa in qa_pairs]

    def _failed_run(self, state: RunState)-> Dict[str, Any]:
        return {"aggregated_metrics": {}}


    # per batch (subgraph) nodes
    def _retrieve(self, qs: QueryState) -> Dict[str, Any]:
        t0 = _now_ms()
        attempt = qs.get("retry_count", {}).get(Node.RETRIEVE, 0) + 1

        try:
            chunks = self.rag_system.retrieve(qs["question"], k=self.k)
            elapsed = _now_ms() - t0
            latency = {**qs.get("latency_ms", {}), Node.RETRIEVE: elapsed}
            self._log(logging.INFO, f"retrieve ok: {len(chunks)} chunks", stage=Node.RETRIEVE,
                      status=LOG_OK, query_id=qs.get("qa_id"), attempt=attempt, latency_ms=elapsed)
            return {"chunks": chunks, QUERY_STATUS: QueryStatus.RETRIEVED, "latency_ms": latency}

        except Exception as e:
            retry_count = dict(qs.get("retry_count", {}))
            retry_count[Node.RETRIEVE] = retry_count.get(Node.RETRIEVE, 0) + 1

            self._log(logging.WARNING, f"retrieve failed (attempt {attempt}/{MAX_RETRIES})",
                      stage=Node.RETRIEVE, status=LOG_FAILED, query_id=qs.get("qa_id"),
                      attempt=attempt, latency_ms=_now_ms() - t0, error=repr(e))

            return {"retry_count": retry_count, QUERY_STATUS: QueryStatus.RETRIEVE_FAILED,
                    "last_error": repr(e)}

    def _route_after_retrieve(self, qs:QueryState)-> str:
        if qs.get(QUERY_STATUS) == QueryStatus.RETRIEVED:
            return Node.GENERATE if self.needs_generation else Node.JUDGE

        if qs.get("retry_count", {}).get(Node.RETRIEVE, 0) >= MAX_RETRIES:
            return Node.FAILED_QUERY

        return Route.RETRY

    def _generate(self, qs:QueryState)-> Dict[str, Any]:
        t0 = _now_ms()
        attempt = qs.get("retry_count", {}).get(Node.GENERATE, 0) + 1

        try:
            result = self.rag_system.generate(qs["question"], qs["chunks"])
            elapsed = _now_ms() - t0
            latency = {**qs.get("latency_ms", {}), Node.GENERATE: elapsed}
            self._log(logging.INFO, "generate ok", stage=Node.GENERATE, status=LOG_OK,
                      query_id=qs.get("qa_id"), attempt=attempt, latency_ms=elapsed)
            return {"generation": result, QUERY_STATUS: QueryStatus.GENERATED, "latency_ms": latency}
        except Exception as e:
            retry_count = dict(qs.get("retry_count", {}))
            retry_count[Node.GENERATE] = retry_count.get(Node.GENERATE, 0) + 1
            self._log(logging.WARNING, f"generate failed (attempt {attempt}/{MAX_RETRIES})",
                      stage=Node.GENERATE, status=LOG_FAILED, query_id=qs.get("qa_id"),
                      attempt=attempt, latency_ms=_now_ms() - t0, error=repr(e))
            return {"retry_count": retry_count, QUERY_STATUS: QueryStatus.GENERATE_FAILED,
                    "last_error": repr(e)}

    def _route_after_generate(self, qs: QueryState) -> str:
        if qs.get(QUERY_STATUS) == QueryStatus.GENERATED:
            return Node.JUDGE
        if qs.get("retry_count", {}).get(Node.GENERATE, 0) >= MAX_RETRIES:
            return Node.FAILED_QUERY
        return Route.RETRY


    def _judge(self, qs: QueryState) -> Dict[str, Any]:
        t0 = _now_ms()
        qa_id = qs.get("qa_id")
        try:
            row_data = {
                "retrieved_ids": [c.source_passage_id for c in qs.get("chunks") or []],
                "gold_ids": qs["gold_ids"],
                "question": qs["question"],
                "gold_answers": qs["gold_answers"],
            }

            if self.needs_generation:
                generation = qs["generation"]
                row_data["answer"] = generation.answer
                row_data["context"] = qs["chunks"]

            metric_results = {}
            metric_errors = []
            for metric in self.metrics:
                try:
                    kwargs = {field: row_data[field] for field in metric.required_inputs}
                    metric_results[metric.name] = metric.score(**kwargs)
                except Exception as exc:
                    self._log(logging.WARNING, f"judge: metric {metric.name} crashed", stage=Node.JUDGE,
                              status=LOG_METRIC_FAILED, query_id=qa_id,
                              error=f"{metric.name}: {exc!r}", exc_info=True)
                    metric_errors.append(f"{metric.name}: {exc!r}")

            if self.metrics and not metric_results:
                # nothing scored -> same as a whole-judge crash (row FAILED, handled below)
                raise RuntimeError("all metrics failed: " + "; ".join(metric_errors))

            elapsed = _now_ms() - t0
            latency = {**qs.get("latency_ms", {}), Node.JUDGE: elapsed}
            update = {
                "metric_results": metric_results,
                QUERY_STATUS: QueryStatus.JUDGED,
                "latency_ms": latency,
            }

            if metric_errors:
                update["last_error"] = "; ".join(metric_errors)

            self._log(logging.INFO, f"judge ok: {len(metric_results)}/{len(self.metrics)} metrics scored",
                      stage=Node.JUDGE, status=LOG_PARTIAL if metric_errors else LOG_OK, query_id=qa_id,
                      latency_ms=elapsed, error=update.get("last_error"))

            update[PER_QUERY_RESULT] = [_snapshot(qs, update)]
            return update

        except Exception as e:
            self._log(logging.ERROR, "judge failed, row marked failed", stage=Node.JUDGE, status=LOG_FAILED,
                      query_id=qa_id, latency_ms=_now_ms() - t0, error=repr(e), exc_info=True)
            update = {QUERY_STATUS: QueryStatus.FAILED, FAILURE_MODE: FailureMode.FAILED,
                    "last_error": repr(e)}
            update[PER_QUERY_RESULT] = [_snapshot(qs, update)]
            return update


    def _failed_query(self, qs: QueryState)-> Dict[str, Any]:
        # Terminla node for branch with exhausted retries
        self._log(logging.ERROR, f"query failed after {MAX_RETRIES} attempts", stage=Node.FAILED_QUERY,
                  status=LOG_FAILED, query_id=qs.get("qa_id"), error=qs.get("last_error"))
        update = {QUERY_STATUS: QueryStatus.FAILED, FAILURE_MODE: FailureMode.FAILED}
        update[PER_QUERY_RESULT] = [_snapshot(qs, update)]

        return update

    # fan-in / downstream (parent graph)
    def _aggregate(self, state: RunState) -> Dict[str, Any]:
        t0 = _now_ms()
        rows = state.get(PER_QUERY_RESULT, [])

        configured_metric_names = {m.name for m in self.metrics}

        aggregates = {}

        for name in configured_metric_names:
            values = []

            for row in rows:
                if row.get(QUERY_STATUS) == QueryStatus.JUDGED and name in row.get("metric_results", {}):
                    values.append(row["metric_results"][name].value)

            finite = [v for v in values if v==v]    # drop NaN

            aggregates[name] = sum(finite)/len(finite) if finite else float("nan")


        # Latency calculations
        for stage in LATENCY_STAGES:
            stage_values = [
                row["latency_ms"][stage]
                for row in rows
                if row.get("latency_ms") and stage in row["latency_ms"]
            ]
            aggregates[f"latency_p50_{stage}"] = _percentile(stage_values, 50)
            aggregates[f"latency_p95_{stage}"] = _percentile(stage_values, 95)

        total_latency_values = [
            sum(row["latency_ms"].values())
            for row in rows
            if row.get("latency_ms")
        ]
        aggregates["latency_p50_total"] = _percentile(total_latency_values, 50)
        aggregates["latency_p95_total"] = _percentile(total_latency_values, 95)


        # Failure mode tagging

        for row in rows:
            if row.get(QUERY_STATUS) == QueryStatus.FAILED:
                continue

            mr = row.get("metric_results", {})

            # a judged row missing a configured metric is incomplete -> judge_failed overrides every other tag
            if configured_metric_names - set(mr):
                row[FAILURE_MODE] = FailureMode.JUDGE_FAILED
                continue

            if RecallAtK.name in mr and math.isclose(mr[RecallAtK.name].value, 0.0, abs_tol=1e-9):
                row[FAILURE_MODE] = FailureMode.RETRIEVAL_MISS

            elif(Faithfulness.name in configured_metric_names
                 and Faithfulness.name in mr
                 and mr[Faithfulness.name].value == mr[Faithfulness.name].value # Check not NaN
                 and mr[Faithfulness.name].value < FAITHFULNESS_LOW_THRESHOLD):

                row[FAILURE_MODE] = FailureMode.LOW_FAITHFULNESS

            elif (
                TokenPRF1.name in configured_metric_names
                and TokenPRF1.name in mr
                and mr[TokenPRF1.name].value == mr[TokenPRF1.name].value  # not NaN
                and mr[TokenPRF1.name].value < TOKEN_PRF1_LOW_THRESHOLD
            ):
                row[FAILURE_MODE] = FailureMode.LOW_CORRECTNESS

            else:
                row[FAILURE_MODE] = FailureMode.PASS

        n_failed = sum(1 for row in rows if row.get(QUERY_STATUS) == QueryStatus.FAILED)
        self._log(logging.INFO, f"aggregate: {len(rows)} rows, {len(rows) - n_failed} judged, {n_failed} failed",
                  stage=Node.AGGREGATE, status=LOG_OK, latency_ms=_now_ms() - t0)
        return {"aggregated_metrics": aggregates}

    def _persist(self, state: RunState) -> Dict[str, Any]:
        """Write this run's per query results, aggregated metrics
        and final run status.

        Chunks are not included
        """
        t0 = _now_ms()
        run_id = state.get("run_id")

        def _generation_answer(row: Dict[str, Any])->Any:
            generation = row.get(GENERATION)
            return generation.answer if generation is not None else None

        query_results = [
            QueryResult(
                run_id=run_id,
                qa_id=row[QA_ID],
                question=row.get(QUESTION),
                query_status=row.get(QUERY_STATUS),
                failure_mode=row.get(FAILURE_MODE),
                last_error=row.get(LAST_ERROR),
                retrieved_ids=[c.source_passage_id for c in row.get(CHUNKS) or []],
                gold_ids=row.get(GOLD_IDS, []),
                generation_answer=_generation_answer(row),
                metric_results=row.get(METRIC_RESULTS, {}),
                latency=row.get(LATENCY_MS, {}),
                category=row.get(CATEGORY),
            ) for row in state.get(PER_QUERY_RESULT, [])
        ]

        if query_results:
            self._repo.save_query_results(query_results)

        self._repo.save_metrics(run_id, state.get("aggregated_metrics", {}))
        self._repo.update_run_status(run_id, state.get(RUN_STATUS), state.get(RUN_STATUS_DETAILS))

        scores = {name: round(value, 4) for name, value in (state.get("aggregated_metrics") or {}).items()
                  if not name.startswith("latency_") and isinstance(value, float) and value == value}
        self._log(logging.INFO, f"persist: {len(query_results)} rows written to {self.db_path}; metrics {scores}",
                  stage=Node.PERSIST, status=str(state.get(RUN_STATUS)), latency_ms=_now_ms() - t0,
                  error=state.get(RUN_STATUS_DETAILS))
        return {}

    def _compare(self, state: RunState) -> Dict[str, Any]:
        # TODO Compare the result of this run with previous runs
        return {}


    # Access point of pipeline

    def run(self, qa_pairs: List[QAPair], run_id:str=None, is_dev:bool=True, passages=None)-> Dict[str, Any]:

        """Built via from_config: the run row stores the full validated RunConfig dump plus its config_hash

        """
        run_id = run_id or str(uuid.uuid4())
        self._run_id = run_id

        if self._run_config is not None:
            is_dev = self._run_config.is_dev
            persisted_config = self._run_config.model_dump(mode="json")
            config_hash = self._run_config.config_hash()
        else:
            persisted_config = {"is_dev": is_dev, "k": self.k}
            config_hash = None



        initial_state: RunState ={
            "run_id": run_id,
            "config": {"is_dev": is_dev, "k": self.k, "passages": passages},
            "corpus_loaded": False,
            "index_built": False,
            "qa_pairs": qa_pairs,
            RUN_STATUS: None,
            RUN_STATUS_DETAILS: None,
            "per_query_results": [],
            "aggregated_metrics": {},
        }

        # Create a run row before graph.invoke()
        # Make it exist in DB even if fail on 1st step
        self._repo.create_run(Run(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_dev=is_dev,
            k=self.k,
            max_workers=self.max_workers,
            config=persisted_config,
            config_hash=config_hash
        ))

        self._log(logging.INFO, f"run started: {len(qa_pairs)} queries, k={self.k}, "
                  f"max_workers={self.max_workers}, config_hash={(config_hash or 'none')[:12]}",
                  stage=RUN_STAGE, status=LOG_STARTED)

        self._start_progress(len(qa_pairs)) # progress bar callback will be called after each row is done

        config = {"max_concurrency": self.max_workers}

        t0 = _now_ms()
        try:
            final_state = self.graph.invoke(initial_state, config=config)
        except Exception as e:
            self._log(logging.ERROR, "run crashed", stage=RUN_STAGE, status=LOG_CRASHED,
                      latency_ms=_now_ms() - t0, error=repr(e), exc_info=True)
            raise

        self._log(logging.INFO, "run finished", stage=RUN_STAGE, status=str(final_state.get(RUN_STATUS)),
                  latency_ms=_now_ms() - t0, error=final_state.get(RUN_STATUS_DETAILS))
        return final_state