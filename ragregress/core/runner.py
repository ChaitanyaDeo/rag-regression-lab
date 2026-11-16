from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


from ragregress.benchmark.schemas import Passage
from ragregress.core.config import DATA_DIR, RunConfig
from ragregress.core.errors import ConfigValidationError, RunFailedError
from ragregress.core.logging import run_log_dir, run_log_file
from ragregress.evaluation.judge import build_judge, resolve_judge
from ragregress.rag.registry import build_rag_system
from ragregress.core.errors import (ConfigValidationError, RunFailedError, PipelineContractError,)
from ragregress.rag.preflight import run_preflight

import logging

logger = logging.getLogger(__name__)  

PASSAGES_PATH = DATA_DIR / "passages.json"
HEALTHY = "healthy"

@dataclass
class RunOutcome:
    run_id: str
    config_hash: str
    final_state: Dict[str, Any]
    elapsed_s: float
    indexed: Optional[int]  # chunks embedded; 0 = index already current; None = pipeline has no ensure_indexed
    log_path: Optional[Path] = None     # per-run JSON log (artifacts/logs/<run_id>.jsonl next to the db)
    preflight_warnings: List[str] = field(default_factory=list)

@dataclass
class CheckOutcome:
    preflight: Any
    indexed: Optional[int]
    determinism: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sampled: int = 0
    raised: int = 0

def default_run_id(run_config: RunConfig, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{run_config.eval_split}-{run_config.config_hash()[:8]}-{now.strftime('%Y%m%d-%H%M%S')}"

def _read_json(path: Path, what: str) -> Any:
    if not path.exists():
        raise ConfigValidationError(
            f"{what} not found (build the data first: python -m ragregress.benchmark.loader)",
            source=str(path)
        )
    return json.loads(path.read_text(encoding="utf-8"))

def load_passages(path: Path = PASSAGES_PATH) -> List[Passage]:
    return [Passage(**row) for row in _read_json(path, "Passages")]

def load_qa_pairs(path: Path):
    from ragregress.benchmark.schemas import QAPair
    return [QAPair(**row) for row in _read_json(path, "split file")]


def build_metrics(run_config: RunConfig, names: Optional[Sequence[str]] = None) -> list:
    # from ragregress.evaluation.metrics import MRR, ExactMatch, Faithfulness, NDCGAtK, PrecisionAtK, RecallAtK, TokenPRF1

    """The metric objects for one run: `names`, or the default-enabled metrics when None."""
    from ragregress.evaluation.metrics.selection import default_metric_names, resolve_metrics

    def get_judge():
        # Build only if selected metric needs it. Provider and model come from the run config;
        # Provider and model come from the run config; endpoint and key from the environment.
        return build_judge(run_config.judge_provider, run_config.judge_model,
                           seed=run_config.judge_seed, max_retries=3)


    return resolve_metrics(default_metric_names() if names is None else names, top_k=run_config.top_k, get_judge=get_judge)

def build_pipeline(run_config: RunConfig, rag_system, metrics):
    from ragregress.pipeline.pipeline import Pipeline

    return Pipeline.from_config(run_config, rag_system=rag_system, metrics=metrics)


# ---------------------------- Preflight check

def _run_preflight(rag_system, qa_pairs, k, corpus_ids=None)->List[str]:
    """Validate the pipeline against the first evaluation question."""

    if not qa_pairs:
        return []

    if corpus_ids is None:
        try:
            passages = load_passages()
            corpus_ids = {p.passage_id for p in passages}
        except ConfigValidationError:
            corpus_ids = None

    result = run_preflight(rag_system, qa_pairs[:1], k, corpus_ids)

    errors = [finding.message for finding in result.findings if finding.severity == "error"]

    if errors:
        raise PipelineContractError(
            "Pipeline contract validation failed:\n- "
            + "\n- ".join(errors)
        )

    warnings = [finding.message for finding in result.findings if finding.severity =="warning"]
    if warnings:
        logger.warning(warnings)
    
    return warnings


def check_pipeline(run_config: RunConfig, samples:int = 3, reindex:bool = False) -> CheckOutcome:
    if not 1<=samples<=20:
        raise ConfigValidationError(f"samples must be between 1 and 20; given {samples}")

    qa_pairs = load_qa_pairs(run_config.split_path())

    if not qa_pairs:
        raise ConfigValidationError("No QA pairs found")

    qa_pairs = qa_pairs[:samples]

    rag_system = build_rag_system(run_config.pipeline.identity, run_config.pipeline.params)

    indexed = None
    corpus_ids = None

    if hasattr(rag_system, "ensure_indexed"):
        passages = load_passages()
        indexed = rag_system.ensure_indexed(passages, force=reindex)
        corpus_ids = {p.passage_id for p in passages}

    preflight = run_preflight(
        rag_system, qa_pairs, run_config.top_k, corpus_ids
    )

    errors = [
        f.message for f in preflight.findings
        if f.severity == "error"
    ]

    warnings = [ f.message for f in preflight.findings if f.severity == "warning"]

    raised = 0
    for qa in qa_pairs:
        try:
            passages = rag_system.retrieve(qa.question, run_config.top_k)
            rag_system.generate(qa.question, passages)
        except Exception as exc:
            raised += 1
            errors.append(str(exc))

    # Repeat first question to check determinism.
    deterministic = False

    try:
        ques = qa_pairs[0].question

        p1 = rag_system.retrieve(ques, run_config.top_k)
        a1 = rag_system.generate(ques,p1)

        p2 = rag_system.retrieve(ques, run_config.top_k)
        a2 = rag_system.generate(ques, p2)

        ids1 = [p.source_passage_id for p in p1]
        ids2 = [p.source_passage_id for p in p2]

        deterministic = ids1 == ids2 and a1.answer == a2.answer
        if not deterministic:
            warnings.append("Retrieval IDs or answers differ")
            

    except Exception as exc:
        warnings.append(f"Repeat check failed: {exc}")
        logger.exception("Repead Check Failed with error:%s",exc, exc_info=True)

    return CheckOutcome(
        preflight=preflight,
        indexed=indexed,
        determinism=deterministic,
        errors=errors,
        warnings=warnings,
        sampled=len(qa_pairs),
        raised=raised,
    )


def execute(
        run_config: RunConfig,
        run_id: Optional[str] = None,
        reindex: bool = False,
        on_row_done: Optional[Callable[[int, int, float], None]] = None,
        metrics: Optional[Sequence[str]] = None,
) -> RunOutcome:
    """Run one experiment end to end and return its outcome.

    `metrics` = names of the metrics to compute (None = the default-enabled ones).

    Raises ConfigValidationError (missing data files, metric names, judge settings), PipelineContractError

    (bad pipeline), RunFailedError (run_status != healthy).

    `on_row_done(done, total, elapsed_s)` is called once with done=0 when the
    run starts and again each time a row finishes (progress reporting).

    The pipeline's INFO+ records for this run are appended to
    <db folder>/artifacts/logs/<run_id>.jsonl (JSON lines, redacted), also when the run fails.
    """
    run_id = run_id or default_run_id(run_config)

    # Fail fast on missing data before loading any model.
    qa_pairs = load_qa_pairs(run_config.split_path())
    # resolve_judge(run_config.judge_provider, run_config.judge_model)       # e.g. a missing API key: fail now, not after indexing

    from ragregress.evaluation.metrics.selection import default_metric_names, needs_judge, validate_metric_names
    names = default_metric_names() if metrics is None else validate_metric_names(metrics)

    if needs_judge(names):                  # a run with no judged metric needs no judge settings
        resolve_judge(run_config.judge_provider, run_config.judge_model)   # e.g. a missing API key: fail now, not after indexing

    rag_system = build_rag_system(run_config.pipeline.identity, run_config.pipeline.params)

    indexed: Optional[int] = None
    corpus_ids = None

    if hasattr(rag_system, "ensure_indexed"):
        passages = load_passages()

        indexed = rag_system.ensure_indexed(passages, force=reindex)

        corpus_ids = {p.passage_id for p in passages}

    preflight_warnings = _run_preflight(rag_system=rag_system, qa_pairs=qa_pairs, k=run_config.top_k, corpus_ids=corpus_ids,)


    metric_objects = build_metrics(run_config, names)
    pipeline = build_pipeline(run_config, rag_system, metric_objects)

    pipeline.on_row_done = on_row_done

    with run_log_file(run_id, run_log_dir(run_config.db_path)) as log_path:
        start = time.monotonic()
        final_state = pipeline.run(qa_pairs=qa_pairs, run_id=run_id)
        elapsed = time.monotonic() - start

    status = final_state.get("run_status")
    if str(getattr(status, "value", status)) != HEALTHY:
        raise RunFailedError(run_id, final_state.get("run_status_detail"))

    return RunOutcome(
        run_id=run_id,
        config_hash=run_config.config_hash(),
        final_state=final_state,
        elapsed_s=elapsed,
        indexed=indexed,
        log_path=log_path,
        preflight_warnings=preflight_warnings,
    )