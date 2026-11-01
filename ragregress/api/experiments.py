"""
ragregress/api/experiments.py   --- YOUR FILE (B3 steps 2-5)

The endpoints for runs, compare and reports. Build it as a function that returns an APIRouter:

    build_router(settings: ApiSettings, jobs: JobManager) -> APIRouter

main.create_app calls it and includes the router (a closure over `settings` and `jobs`; FastAPI's `Depends` is the
other way to do this, a good exercise for later).

The tests REPLACE these three names on this module with monkeypatch, so import them at the top and call them
through the module's own names (do not alias them, do not import them inside the functions):
    from ragregress.core.runner import default_run_id
    from ragregress.evaluation.comparator import compare_runs
    from ragregress.report import generate_report        # also: render_html, TargetsFileError

Step 2  GET /runs (list, newest first) and GET /runs/{run_id}          -k "startup or runs_are or run_list or run_detail or test_unknown_run"
Step 3  POST /compare                                                  (see the test file: "compare" group)
Step 4  GET /reports/{pair}                                            ("report" group)
Step 5  POST /runs                                                     ("post_runs", "config", ... group)

Status codes (the spec):
  POST /runs      202 started | 404 no such config | 409 a run is in progress / the run id exists | 422 bad config
  GET  /runs/{id} 200 | 404
  POST /compare   200 (even for FAIL / INCONCLUSIVE: the verdict is in the body) | 404 unknown baseline | 422 bad request
                  A problem with ONE candidate is that entry's `error`; the other candidates are still compared.
  GET  /reports   200 HTML | 404 unknown run | 400 runs that cannot be compared | 422 malformed id | 500 broken targets file
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from typing import List, Optional
import logging 
import re

from ragregress.api.jobs import Job, JobBusy, JobExists, JobManager
from ragregress.api.settings import ApiSettings
from ragregress.core.runner import default_run_id
from ragregress.evaluation.comparator import compare_runs, MISSING
from ragregress.core.logging import redact
from ragregress.core.errors import ComparisonInputError, ConfigError, RagRegressError
from ragregress.core.config import load_run_config
from ragregress.report import TargetsFileError, generate_report, render_html
from ragregress.evaluation.judge import resolve_judge

from ragregress.storage.repositories import Repository
from ragregress.storage.models import Run
from ragregress.api.schemas import (RunSummary, RunDetail, ComparisonOut, RunRef, 
                                    ValidityOut,ConfigChangeOut, MetricOut, clean_number, CompareResponse,
                                    CompareRequest, CompareEntry, ID_PATTERN, JobView, NAME_PATTERN, RunAccepted, RunRequest)

from ragregress.evaluation.regression import assess_regression, Validity, RegressionReport, ComparisonResult

log = logging.getLogger(__name__)
REPORT_SEPARATOR = "__vs__"
_ID = re.compile(ID_PATTERN)

def _to_runref(snapshot):
    run = snapshot.run
    cfg = run.config or {}
    return RunRef(
        run_id=run.run_id,
        config_hash=run.config_hash,
        eval_split=cfg.get("eval_split"),
        top_k=run.k,
        legacy="pipeline" not in cfg,
    )

def _to_validityOut(v: Validity) -> ValidityOut:
    return ValidityOut(measured=v.measured, attempted=v.attempted, ratio=v.ratio)

def _comparison_out(result: ComparisonResult, report: RegressionReport) -> ComparisonOut:
    
    return ComparisonOut(
        verdict= report.verdict.value,
        exit_code=report.exit_code,
        reasons= list(report.reasons),
        warnings=list(result.warnings),
        gated=list(report.gated),
        baseline = _to_runref(result.baseline),
        candidate= _to_runref(result.candidate),
        validity_baseline= _to_validityOut(report.validity_baseline),
        validity_candidate= _to_validityOut(report.validity_candidate),
        
        config_changes= [
            ConfigChangeOut(  
                key= change.key,
                before= None if change.before is MISSING else change.before,
                after= None if change.after is MISSING else change.after,
                before_missing=change.before is MISSING,
                after_missing =change.after is MISSING
            ) for change in result.config_changes],

        metrics =[
            MetricOut(
                name = m.name,
                status= m.status.value,
                before=clean_number(m.before),
                after=clean_number(m.after),
                delta=clean_number(m.delta),
                n_common=m.n_common,
                better=m.better,
                worse= m.worse,
                tie = m.tie,
                threshold= clean_number(m.threshold),
                higher_is_better=m.higher_is_better,
                note=m.note,
            ) for m in report.metrics
        ]
    )


def _job_view(job:  Optional[Job])-> Optional[JobView]:
    if job is None:
        return None
    return JobView(
        state = job.state,
        started_at = job.started_at,
        finished_at = job.finished_at,
        rows_done = job.rows_done,
        rows_total = job.rows_total,
        error= job.error
    )

def build_router(settings: ApiSettings, jobs: JobManager) -> APIRouter:
    router = APIRouter()

    repo = Repository(settings.db_path)

    def run_to_run_summary(run: Run) -> RunSummary:
        pipeline = run.config.get("pipeline")
        pipeline_id = (
            pipeline.get("id")
            if isinstance(pipeline, dict)
            else None
        )
        run_summary = RunSummary(
            run_id=run.run_id,
            created_at= run.created_at,
            is_dev = run.is_dev,
            k = run.k,
            run_status=run.run_status,
            config_hash = run.config_hash,
            eval_split = run.config.get("eval_split"),
            pipeline_id = pipeline_id,
        )
        return run_summary

    @router.get("/runs", response_model = List[RunSummary])
    def list_runs():
        runs = repo.list_runs()        
        return [run_to_run_summary(run) for run in runs]


    @router.get("/runs/{run_id}", response_model=RunDetail)
    def run_detail(run_id: str):

        run = repo.get_run(run_id)
        job = jobs.get(run_id)

        if run is None and job is None:
            raise HTTPException(status_code=404, detail="Run not found")

        elif run is None:
            # Place holder RunDetail for run that is accepted but not yet in db
            return RunDetail(
                run_id = job.run_id,
                created_at=job.started_at,
                is_dev= None, k = None,
                run_status =None, eval_split=None, pipeline_id=None, 
                config_hash=None, run_status_detail=None,
                config={}, metrics={},
                job=_job_view(job)
            )
        
        summary = run_to_run_summary(run)

        run_status_detail = run.run_status_detail
        if run_status_detail:
            run_status_detail = redact(run_status_detail)

        metrics = repo.get_metrics(run_id)
        metric_dict = {
            m.metric_name: m.value for m in metrics
        }
        
        return RunDetail(
            **summary.model_dump(),
            run_status_detail=run_status_detail,
            config= run.config,
            metrics=metric_dict,
            job =_job_view(job)
        )


    # Compare
    @router.post("/compare", response_model=CompareResponse)
    def compare(body: CompareRequest):

        if repo.get_run(body.baseline) is None:
            raise HTTPException(status_code=404, detail=f"No run with id {body.baseline!r}")

        results = []
        for c in body.candidates:
            try:
                result = compare_runs(repo, body.baseline, c)
                report = assess_regression(result)

                entry = CompareEntry(
                    candidate=c, ok=True, error=None,
                    comparison=_comparison_out(result, report),
                )

            except RagRegressError as e:
                # input problem: unknown id, dev run, different split, ...
                entry = CompareEntry(candidate=c, ok=False, error=redact(str(e)),comparison=None)

            except Exception as e:
                # a bug in one comparison must not hide the other candidates
                log.exception("compare %s vs %s crashed", body.baseline, c)
                entry = CompareEntry(candidate=c, ok=False, error=redact(f"unexpected {type(e).__name__}: {e}"),comparison=None)
            
            results.append(entry)
                
        return CompareResponse(baseline=body.baseline, results= results)


    # Reports
    @router.get("/reports/{pair}", response_class=HTMLResponse)
    def get_report(pair: str) -> HTMLResponse:
        name = pair[:-5] if pair.endswith(".html") else pair

        baseline, sep, candidate = name.partition(REPORT_SEPARATOR)

        if not sep or _ID.fullmatch(baseline) is None or _ID.fullmatch(candidate) is None or REPORT_SEPARATOR in candidate:
            raise HTTPException(status_code=422, detail = "report id must look like <baseline>__vs__<candidate>")

        if repo.get_run(baseline) is None or repo.get_run(candidate) is None:
            raise HTTPException(status_code=404, detail = f"Unknown run id in {pair!r}")

        try:
            ctx = generate_report(repo, baseline, candidate,
                              passages_path=settings.passages_path,
                              splits_dir=settings.splits_dir,
                              top = settings.top,
                              targets_path = settings.targets_file())

        except TargetsFileError as e:
            raise HTTPException(status_code=500, detail=redact(str(e)))
        
        except ComparisonInputError as e:
            raise HTTPException(status_code = 400, detail=redact(str(e)))


        return HTMLResponse(render_html(ctx))

    @router.post("/runs", status_code=202, response_model=RunAccepted)
    def start_run(body: RunRequest)->RunAccepted:
        """Start a run. The config name is name of yaml file in configs_dir"""

        path = Path(settings.configs_dir) / f"{body.config}.yaml"

        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no config named {body.config!r}")

        try:
            config = load_run_config(path)
            resolve_judge(config.judge_provider, config.judge_model)
        except ConfigError as e:
            raise HTTPException(status_code = 422, detail = redact(str(e)))

        if Path(config.db_path).resolve() != Path(settings.db_path).resolve():
            raise HTTPException(status_code=422, detail=f"Run database db_path must match the API database")

        
        run_id = default_run_id(config)     # Generate run ID and check for duplicates

        if jobs.knows(run_id) or repo.get_run(run_id) is not None:
            raise HTTPException(status_code=409, detail=f"Run {run_id} already exists")

        # Start a run in background thread

        try:
            job = jobs.start(config, run_id, body.config)

        except JobBusy as e:
             raise HTTPException(status_code=409, detail=f"Another run is in progress: {e.run_id}") from e

        except JobExists as exc:
            raise HTTPException(status_code=409, detail=f"Run already exists: {run_id}") from exc

        return RunAccepted(run_id = job.run_id, state=job.state, status_url = f"/runs/{job.run_id}")

    return router

