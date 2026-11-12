"""
ragregress/api/jobs.py

A run takes tens of minutes, so POST /runs starts it on a background
thread and returns at once; GET /runs/{run_id} shows the thread's progress.

Rules:
    - One run at a time per process. A second start raises JobBusy (the API answers 409)
    - The record live in memory only. After restart the stored run is all that is left; a run
      that was in progress when the process died keeps run_status null there.
    - The thread is a dameon: stopping the server does not wait for a run to finish
    - Error text is redacted before it is stored: it can contain whatever a library put in an exception.

`executor` is injected so tests never load a model: executor(config, run_id, on_row_done) -> None.
When the request chose metrics, it is also called with metrics=<names>; otherwise without that keyword.  
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Sequence

from ragregress.core.config import RunConfig
from ragregress.core.errors import RagRegressError
from ragregress.core.logging import redact

log = logging.getLogger(__name__)

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

ProgressCallback = Callable[[int, int, float], None]
Executor = Callable[[RunConfig, str, ProgressCallback], None]

def execute_run(config: RunConfig, run_id: str, on_row_done: ProgressCallback,
                metrics: Optional[Sequence[str]] = None)->None:
    # Real executor : similar to ragregress run
    from ragregress.core.runner import execute
    execute(config, run_id = run_id, on_row_done=on_row_done, metrics=metrics)


class JobBusy(Exception):
    """Run already in progress"""
    def __init__(self, run_id:str):
        self.run_id = run_id
        super().__init__(run_id)


class JobExists(Exception):
    """This process already started a run with this id."""
 
    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(run_id)


@dataclass
class Job:
    run_id: str
    config_name: str
    started_at: str
    state: str = RUNNING
    finished_at: Optional[str] = None
    rows_done: int = 0
    rows_total: Optional[int] = None
    error: Optional[str] = None

def _now()->str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(self, executor: Executor = execute_run):
        self._executor = executor
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._current: Optional[str] = None

    def knows(self, run_id:str):
        with self._lock:
            return run_id in self._jobs

    def get(self, run_id:str) -> Optional[Job]:
        """A copy, so a reader never sees a half-updated record."""
        with self._lock:
            job = self._jobs.get(run_id)
            return None if job is None else Job(**vars(job))


    def start(self, config: RunConfig, run_id: str, config_name: str,
              metrics: Optional[Sequence[str]] = None) -> Job:
        with self._lock:
            if self._current is not None and self._jobs[self._current].state == RUNNING:
                raise JobBusy(self._current)

            if run_id in self._jobs:
                raise JobExists(run_id)

            job = Job(run_id=run_id, config_name=config_name, started_at=_now())
            self._jobs[run_id] = job
            self._current = run_id

            accepted = Job(**vars(job))               # taken before the thread starts: a very fast run must still

        threading.Thread(target=self._work, args=(config, job, metrics), name=f"run-{run_id}", daemon=True).start()

        return accepted

    def _progress(self, job: Job) -> ProgressCallback:
        def on_row_done(done:int, total:int, elapsed_s:float)->None:
            with self._lock:
                job.rows_done, job.rows_total = done, total
        
        return on_row_done

    def _work(self, config: RunConfig, job:Job, metrics: Optional[Sequence[str]] = None):
        state, error = SUCCEEDED,  None

        try:
            extra = {} if metrics is None else {"metrics": list(metrics)}
            self._executor(config, job.run_id, self._progress(job), **extra)
        except RagRegressError as exc:
            state, error = FAILED, redact(str(exc))
            log.warning("run %s failed: %s", job.run_id, error, extra={"run_id": job.run_id})

        except Exception as e:
            state, error = FAILED, redact(f"unexpected {type(e).__name__}: {e}")
            log.exception("run %s crashed", job.run_id, extra={"run_id": job.run_id})

        with self._lock:
            job.state, job.error, job.finished_at = state, error, _now()