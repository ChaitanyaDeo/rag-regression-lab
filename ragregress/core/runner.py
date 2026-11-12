from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ragregress.benchmark.schemas import Passage
from ragregress.core.config import DATA_DIR, RunConfig
from ragregress.core.errors import ConfigValidationError, RunFailedError
from ragregress.core.logging import run_log_dir, run_log_file
from ragregress.evaluation.judge import build_judge, resolve_judge
from ragregress.rag.registry import build_rag_system


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

    rag_system = build_rag_system(run_config.pipeline.id, run_config.pipeline.params)

    indexed: Optional[int] = None

    if hasattr(rag_system, "ensure_indexed"):
        indexed = rag_system.ensure_indexed(load_passages(), force=reindex)

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
    )