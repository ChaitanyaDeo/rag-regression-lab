"""
ragregress/core/errors.py

Shared exception types: CLI and API catch RagRegressError subclasses
instead of each defining its own error handling

Each class carries its own `exit_code` so the CLI maps an exception to
a process exit code with `exc.exit_code` -- no per-type if/elif.

Exit codes:
    0  run completed (partial per-query failures allowed; Day 8's
       INCONCLUSIVE rule judges quality, not the exit code)
    1  unexpected exception (a bug) -- Python's default
    2  RESERVED: Typer/Click use 2 for CLI usage errors; never reuse
    3  ConfigError and subclasses
    4  PipelineContractError
    5  RunFailedError

    6  compare verdict FAIL                 (a verdict, not an exception)
    7  compare verdict INCONCLUSIVE         (a verdict, not an exception)
    8  ComparisonInputError and subclasses  (unknown run id, is_dev run, incompatible runs)


"""

from __future__ import annotations
 
from typing import List, Optional, Tuple

class RagRegressError(Exception):
    """Base for every error this package raises on purpose."""
    exit_code = 1

class ConfigError(RagRegressError):
    """A run configuration could not be loaded or is invalid."""
    exit_code = 3

class ConfigValidationError(ConfigError):
    """Config file missing, unparseable YAML, or failed RunConfig validation

    errors` is a list of (location, message) pairs, e.g.
    ("pipeline.params.chunk_size", "Field required"), so callers can
    print one clean line per problem.
    """

    def __init__(self, message: str, errors: Optional[List[Tuple[str, str]]] = None, source: Optional[str] = None):
        self.message = message
        self.errors = errors or []
        self.source = source
        super().__init__(message)

    def __str__(self)->str:
        header = f"{self.source}: {self.message}" if self.source else self.message
        if not self.errors:
            return header
        lines = [header] + [f"  - {loc}: {msg}" for loc, msg in self.errors]
        return "\n".join(lines)


class UnknownPipelineError(ConfigError):
    """A pipeline id is not in the registry.

    Inside RunConfig validation an unknown id surfaces as a
    ConfigValidationError instead; this type is for direct registry
    calls (e.g. a future API path) that bypass RunConfig.
    """

class PipelineContractError(RagRegressError):
    """A built object does not satisfy the RAGSystem protocol (DEC-D7-02).
 
    Note: isinstance() on a runtime_checkable Protocol only checks that
    the methods exist, not their signatures or return types.
    """
    exit_code = 4

class RunFailedError(RagRegressError):
    """The pipeline finished with run_status=unhealthy (e.g. Ollama down
    at the Ingest health check). Raised by the CLI after inspecting the
    final state -- the pipeline itself still persists the failed run and
    returns normally.
    """
 
    exit_code = 5
 
    def __init__(self, run_id: str, detail: Optional[str] = None):
        self.run_id = run_id
        self.detail = detail
        msg = f"run {run_id} failed"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class ComparisonInputError(RagRegressError):
    """`ragregress compare` was given inputs it cannot compare
    
    Verdicts (PASS / FAIL / INCONCLUSIVE) are NOT errors: the CLI maps them to
    exit codes 0 / 6 / 7 itself. This family is only for unusable inputs.
    """

    exit_code = 8

class UnknownRunError(ComparisonInputError):
    """No run with this id exists in the database."""
    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"unknown run id: {run_id!r}")

class DevRunRejectedError(ComparisonInputError):
    """A dev-split run (is_dev=1) was passed as baseline or candidate."""

    def __init__(self, run_id: str, role: str):
        self.run_id = run_id
        self.role = role
        super().__init__(
            f"{role} run {run_id!r} is a dev run (is_dev=1); "
            "dev runs cannot be used for regression comparison"
        )

class IncompatibleRunsError(ComparisonInputError):
    """The two runs are not comparable (e.g. different eval_split)."""
    