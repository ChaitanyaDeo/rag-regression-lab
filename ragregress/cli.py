"""
ragregress/cli.py


Thin by design: parse flags, call core.runner.execute(), print, and map
exceptions to exit codes (RagRegressError.exit_code; anything else = 1).
By default only a one-line message is printed; --debug adds the traceback.

Usage: 
    python -m ragregress.cli --config <path> [--run-id <id>] [--reindex] [--debug]
"""

from __future__ import annotations
 
import sys
import traceback
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from typing import Optional
 
import typer
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
from tqdm.contrib.logging import logging_redirect_tqdm
 
from ragregress.core.config import RunConfig, load_run_config
from ragregress.core.errors import RagRegressError
from ragregress.core.runner import PASSAGES_PATH, RunOutcome, default_run_id, execute
from ragregress.core.errors import ComparisonInputError
from ragregress.evaluation.comparator import MISSING, ComparisonResult, RunSnapshot, compare_runs
from ragregress.evaluation.diagnostics import diagnose, render_text
from ragregress.evaluation.regression import RegressionReport, assess_regression
from ragregress.storage.database import DB_PATH_DEFAULT
from ragregress.storage.repositories import Repository

 
app = typer.Typer(add_completion=False, no_args_is_help=True, help="RAG Regression Lab.")


@app.callback()
def _main() -> None:
    """RAG Regression Lab. (This callback keeps `run` a named subcommand.)"""

def _print_plan(cfg: RunConfig, run_id: str) -> None:
    """Print a summary of run config and run_id """
    echo = typer.echo

    echo("dry run: config is valid, nothing was executed")
    echo(f"  pipeline     : {cfg.pipeline.id} {cfg.pipeline.params.model_dump()}")
    echo(f"  top_k        : {cfg.top_k}")
    echo(f"  judge_model  : {cfg.judge_model}")
    echo(f"  eval_split   : {cfg.eval_split} (is_dev={cfg.is_dev})")
    echo(f"  max_workers  : {cfg.max_workers}")
    echo(f"  db_path      : {cfg.db_path}")
    echo(f"  config_hash  : {cfg.config_hash()}")
    echo(f"  run_id       : {run_id}")

    for label, path in (("split file", cfg.split_path()), ("passages", PASSAGES_PATH)):
        echo(f"  {label:<12} : {path} ({'found' if Path(path).is_file() else 'MISSING'})")

def _print_summary(outcome: RunOutcome) -> None:
    echo = typer.echo
    state = outcome.final_state
    rows = state.get("per_query_results") or []

    echo(f"run_id      : {outcome.run_id}")
    echo(f"config_hash : {outcome.config_hash[:12]}")
    echo(f"run_status  : {getattr(state.get('run_status'), 'value', state.get('run_status'))}")
    echo(f"elapsed     : {outcome.elapsed_s:.1f}s ({outcome.elapsed_s / max(len(rows), 1):.1f}s/row)")
    if outcome.indexed is None:
        echo("index       : skipped (pipeline manages its own index)")
    elif outcome.indexed == 0:
        echo("index       : up to date")
    else:
        echo(f"index       : rebuilt ({outcome.indexed} chunks embedded)")

    echo(f"\nfailure modes ({len(rows)} rows):")
    modes = Counter(str(row.get("failure_mode")) for row in rows)

    for mode, count in sorted(modes.items(), key=lambda kv: (kv[0] == "None", kv[0])):
        echo(f"  {mode}: {count}")

    echo("\nmetrics:")
    for name, value in (state.get("aggregated_metrics") or {}).items():
        echo(f"  {name}: {value}")


@contextmanager
def _redirect_prints_through_tqdm():
    """Send print()/stderr output through tqdm.write while a bar is active, so
    the pipeline's own debug prints scroll above the bar instead of tearing it."""
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        with logging_redirect_tqdm():  # logging handlers hold the original stderr
            yield orig_out_err[0]
    finally:
        sys.stdout, sys.stderr = orig_out_err

def _run_with_progress(cfg: RunConfig, run_id: Optional[str], reindex: bool) -> RunOutcome:
    """Run with a tqdm bar pinned to the bottom."""

    bar: Optional[tqdm] = None
    with _redirect_prints_through_tqdm() as orig_stdout:

        def on_row_done(done: int, total: int, elapsed_s: float) -> None:
            nonlocal bar
            if bar is None:
                bar = tqdm(total=total, unit="row", desc="rows", file=orig_stdout, dynamic_ncols=True)
            bar.update(done - bar.n)

        try:
            return execute(cfg, run_id=run_id, reindex=reindex, on_row_done=on_row_done)
        finally:
            if bar is not None:
                bar.close()


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", help="Path to a run config YAML."),
        run_id: Optional[str] = typer.Option(None, "--run-id", help="Override the run id (default: <split>-<hash8>-<timestamp>).",),
        reindex: bool = typer.Option(False, "--reindex", help="Force a full rebuild of this config's Chroma collection."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate the config and print the resolved plan; run nothing."),
        debug: bool = typer.Option(False, "--debug", help="Print a traceback on errors."),

) -> None:
    """Run one evaluation described by a config file and store the result."""

    exit_code = 0
    try:
        cfg = load_run_config(config)
        run_id = run_id or default_run_id(cfg)
        if dry_run:
            _print_plan(cfg, run_id)
        else:
            # _print_summary(execute(cfg, run_id=run_id, reindex=reindex))
            _print_summary(_run_with_progress(cfg, run_id, reindex))

    except RagRegressError as exc:
        typer.echo(f"error: {exc}", err=True)
        if debug:
            traceback.print_exc(file=sys.stderr)
        exit_code = exc.exit_code

    except Exception as exc:
        typer.echo(
            f"error: unexpected {type(exc).__name__}: {exc} (re-run with --debug for a traceback)",
            err=True,
        )
        if debug:
            traceback.print_exc(file=sys.stderr)
        exit_code = 1

    if exit_code:
        raise typer.Exit(exit_code)     # outside the try: Exit is an Exception subclass


# ---------------------------------------------------------------- compare

def _fmt(value: Optional[float], width: int = 9) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.4f}"

def _fmt_delta(value: Optional[float]) -> str:
    return f"{'n/a':>9}" if value is None else f"{value:>+9.4f}"

def _describe_run(label: str, snap: RunSnapshot) -> str:
    run = snap.run
    config_hash = run.config_hash[:12] if run.config_hash else "none (Legacy)"
    split = (run.config or {}).get("eval_split", "?")
    return f"{label:<9}: {run.run_id}  (hash {config_hash}, split {split}, k={run.k})"


def _print_comparison(result: ComparisonResult, report:RegressionReport) -> None:
    echo = typer.echo

    echo(_describe_run("baseline", result.baseline))
    echo(_describe_run("candidate", result.candidate))

    echo("\nConfig changes:")
    if result.config_changes:
        for change in result.config_changes:
            echo(f"  {change.key}: {change.before} -> {change.after}")

        if any(c.before is MISSING or c.after is MISSING for c in result.config_changes):
            echo("  (- = key absent from that run's stored config: legacy run, config incomplete)")

    else:
        echo("  (none)")

    echo("\nWarnings:")
    if result.warnings:
        for warning in result.warnings:
            echo(f"  - {warning}")

    else:
        echo("  (none)")

    vb, vc = report.validity_baseline, report.validity_candidate
    echo(
        f"\nvalidity : baseline {vb.measured}/{vb.attempted} ({vb.ratio:.0%}) "
        f"| candidate {vc.measured}/{vc.attempted} ({vc.ratio:.0%})"
    )
    echo(f"gated    : {', '.join(report.gated) if report.gated else '(none)'}")

    echo(f"\n{'metric':<14}{'before':>9}{'after':>9} {'delta':>9}  {'n':<5}{'b/w/t':<12}{'threshold':<10}status")
    for m in report.metrics:
        threshold = "-" if m.threshold is None else f"{m.threshold:.2f}"
        status = m.status.value + (f" ({m.note})" if m.note else "")
        echo(
            f"{m.name:<14}{_fmt(m.before)}{_fmt(m.after)} {_fmt_delta(m.delta)}  "
            f"{m.n_common:<5}{f'{m.better}/{m.worse}/{m.tie}':<12}{threshold:<10}{status}"
        )

    echo(f"\nVERDICT: {report.verdict.value}")
    for reason in report.reasons:
        echo(f"  - {reason}")


@app.command()
def compare(
    baseline: str = typer.Option(..., "--baseline", help="Run id of the baseline run."),
    candidate: str = typer.Option(..., "--candidate", help="Run id of the candidate run"),
    db: Path = typer.Option(DB_PATH_DEFAULT, "--db", help="SQLite database that holds both runs."),

    worst: int = typer.Option(0,"--worst", min=0, 
                              help="Also list the N worst-regressed queries per gated metric, plus category / failure-mode groups (0 = off).",),
    debug: bool = typer.Option(False, "--debug", help="Print a traceback on errors")
) -> None:
    """Compare two stored runs and print a PASS / FAIL / INCONCLUSIVE verdict.

    Exit codes: 0 PASS, 6 FAIL, 7 INCONCLUSIVE, 8 invalid input (unknown run id,
    dev run, incompatible runs, missing database).
    """

    exit_code = 0

    try:
        if not db.is_file():
            raise ComparisonInputError(f"database not found: {db}")

        result = compare_runs(Repository(db), baseline, candidate)
        report = assess_regression(result)

        _print_comparison(result, report)
        if worst:
            typer.echo("")
            for line in render_text(diagnose(result, report, top_n=worst)):
                typer.echo(line)
                
        exit_code = report.exit_code

    except RagRegressError as exc:
        typer.echo(f"error: {exc}", err=True)
        if debug:
            traceback.print_exc(file=sys.stderr)
        exit_code = exc.exit_code

    except Exception as exc:
        typer.echo(f"error: unexpected {type(exc).__name__}: {exc} (re-run with --debug for a traceback)",
                   err=True)
        if debug:
            traceback.print_exc(file=sys.stderr)
        exit_code = 1

    if exit_code:
        raise typer.Exit(exit_code)     # outside the try: Exit is an Exception subclass


if __name__ == "__main__":
    app()