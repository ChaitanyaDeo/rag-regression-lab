"""
    ragregress/evaluation/comparator.py

    Reads two stored runs and describes how they differ
    It does NOT decide PASS / FAIL / INCONCLUSIVE


"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ragregress.core.errors import (
    DevRunRejectedError,
    IncompatibleRunsError,
    UnknownRunError,
)

from ragregress.storage.models import Run
from ragregress.storage.repositories import Repository

STATUS_JUDGE = "judged"
STATUS_FAILED = "failed"

# Canonical display order; any other metric a run contains is appended alphabetically.
DISPLAY_ORDER = ("recall_at_k", "mrr", "token_prf1", "faithfulness")

# Keys excluded from config_hash, not experiment-defining.
_OPERATIONAL_KEYS = frozenset({"db_path", "max_workers"})

# judge_provider is left out of a stored config while it is the default (RunConfig does that), so a
# stored config without the key means "ollama" -- every run written before the field existed.
_DEFAULT_JUDGE_PROVIDER = "ollama"

_TIE_TOL = 1e-9


class _Missing:
    """Sentinel for 'key absent from this run's config' (rendered as an em dash)."""

    def __repr__(self) -> str:
        return "—"
    __str__ = __repr__

MISSING = _Missing()

class RowKind(str, Enum):
    MEASURED = "measured"
    SYSTEM_FAILURE = "system_failure"        # no answer: retrieve/generate exhausted retries
    EVALUATOR_FAILURE = "evaluator_failure"  # answer exists, judging crashed

@dataclass
class RowView:
    qa_id: str
    category: Optional[str]
    failure_mode: Optional[str]
    kind: RowKind
    # One entry per metric applicable to the run; None = unmeasured for this row.
    values: Dict[str, Optional[float]]

@dataclass(frozen = True)
class RunSnapshot:
    run: Run
    rows: Dict[str, RowView]
    applicable: frozenset  # metric names present in >= 1 judged row

    @property
    def attempted(self)-> int:
        return len(self.rows)


@dataclass(frozen=True)
class ConfigChange:
    key: str       # dotted, e.g. "pipeline.params.chunk_size"
    before: Any    # MISSING if absent from the baseline config
    after: Any     # MISSING if absent from the candidate config

@dataclass(frozen=True)
class MetricDelta:
    name: str
    before: Optional[float]   # mean over common rows; None if no common rows
    after: Optional[float]
    delta: Optional[float]    # after - before
    n_common: int
    better: int               # rows where candidate > baseline
    worse: int
    tie: int
    applicable_base: bool
    applicable_cand: bool

@dataclass(frozen=True)
class ComparisonResult:
    baseline: RunSnapshot
    candidate: RunSnapshot
    config_changes: List[ConfigChange]
    legacy_run_ids: Tuple[str, ...]
    warnings: List[str]
    ungated: Dict[str, str]          # metric -> reason it must not be gated
    metrics: List[MetricDelta]
    common_row_ids: Tuple[str, ...]
    only_in_baseline: Tuple[str, ...]
    only_in_candidate: Tuple[str, ...]


# ----------------------------------------------------------------------------------------

def _metric_value(entry: Any) -> Optional[float]:
    """Stored form is {"value": x, "reason": ...}; NaN / non-numeric -> None."""
    value = entry.get("value") if isinstance(entry, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if math.isnan(value) else float(value)


def load_snapshot(repo: Repository, run_id: str) -> RunSnapshot:
    run = repo.get_run(run_id)

    if run is None:
        raise UnknownRunError(run_id)

    results = repo.get_query_results(run_id)

    applicable = frozenset(
        name for r in results if r.query_status == STATUS_JUDGE for name in (r.metric_results or {})
    )

    rows: Dict[str, RowView] = {}
    for r in results:
        if r.query_status == STATUS_JUDGE:
            kind = RowKind.MEASURED
            mr = r.metric_results or {}
            values = {m: _metric_value(mr.get(m)) for m in applicable}

        elif r.query_status == STATUS_FAILED and r.generation_answer is None:
            kind = RowKind.SYSTEM_FAILURE
            values = {m: 0.0 for m in applicable}

        else:
            # failed WITH an answer = judge crash.
            kind = RowKind.EVALUATOR_FAILURE
            values = {m: None for m in applicable}

        rows[r.qa_id] = RowView(
            qa_id=r.qa_id,
            category=r.category,
            failure_mode=r.failure_mode,
            kind=kind,
            values=values,
        )


    return RunSnapshot(run=run, rows=rows, applicable=applicable)

# ----------------------------------------------------------------- Guardrail for compaing runs

def _legacy_run_ids(*snapshots: RunSnapshot) -> Tuple[str, ...]:
    # Runs built via RunConfig always carry `pipeline`; legacy runs only have {is_dev, k}.
    return tuple(s.run.run_id for s in snapshots if "pipeline" not in (s.run.config or {}))

def check_comparable(base: RunSnapshot, cand: RunSnapshot) -> Tuple[List[str], Dict[str, str]]:
    """Return (warnings, ungated). Raises ComparisonInputError subclasses (exit 8)."""

    for role, snap in (("baseline", base), ("candidate", cand)):
        if snap.run.is_dev:
            raise DevRunRejectedError(snap.run.run_id, role)

    warnings: List[str] = []
    ungated: Dict[str, str] = {}
    bcfg, ccfg = base.run.config or {}, cand.run.config or {}

    legacy = _legacy_run_ids(base, cand)

    if legacy:
        # Legacy runs have no config_hash, so cannot be gated
        warnings.append(
            f"legacy run: config incomplete ({', '.join(legacy)});"
            "eval_split / judge_model comparability could not be fully verified"
        )


    b_split, c_split = bcfg.get("eval_split"), ccfg.get("eval_split")

    if b_split is not None and c_split is not None and b_split != c_split:
        raise IncompatibleRunsError(
            f"eval_split differs (baseline={b_split!r}, candidate={c_split!r})"
        )

    if base.run.k != cand.run.k:
        reason = f"top_k differs (baseline={base.run.k}, candidate={cand.run.k})"
        warnings.append(f"{reason}; recall_at_k is not gated")
        ungated["recall_at_k"] = reason


    # The judge is the pair (judge_provider, judge_model): the same model name on a different service
    # is a different judge. Only judged when both runs name a judge_model (legacy runs do not).
    b_judge, c_judge = bcfg.get("judge_model"), ccfg.get("judge_model")
    if b_judge is not None and c_judge is not None:
        b_provider = bcfg.get("judge_provider") or _DEFAULT_JUDGE_PROVIDER
        c_provider = ccfg.get("judge_provider") or _DEFAULT_JUDGE_PROVIDER
        reasons: List[str] = []
        if b_judge != c_judge:
            reasons.append(f"judge_model differs (baseline={b_judge!r}, candidate={c_judge!r})")
        if b_provider != c_provider:
            reasons.append(f"judge_provider differs (baseline={b_provider!r}, candidate={c_provider!r})")
        if reasons:
            reason = "; ".join(reasons)
            warnings.append(f"{reason}; faithfulness is not gated")
            ungated["faithfulness"] = reason

    only_b = set(base.rows) - set(cand.rows)
    only_c = set(cand.rows) - set(base.rows)

    if only_b or only_c:
        warnings.append(
            f"question sets differ ({len(only_b)} only in baseline, {len(only_c)} only in "
            f"candidate); metrics use the {len(set(base.rows) & set(cand.rows))} common rows"
        )

    return warnings, ungated


# ---------------------------------------------------------------- diffs

def _flatten(obj: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict into dotted keys"""
    flat= {}

    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value

    return flat

def _normalise_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Flatten a config and remove operational keys"""
    out = {k: v for k, v in (cfg or {}).items() if k not in _OPERATIONAL_KEYS}
    if "k" in out and "top_k" not in out:  # legacy configs call it `k`
        out["top_k"] = out.pop("k")
    return _flatten(out)


def diff_config(base_cfg: Optional[Mapping[str, Any]], cand_cfg: Optional[Mapping[str, Any]]) -> List[ConfigChange]:
    """Return a list of ConfigChanges for key where config differs"""
    a, b = _normalise_config(base_cfg), _normalise_config(cand_cfg)
    changes = []
    for key in sorted(set(a) | set(b)):
        before, after = a.get(key, MISSING), b.get(key, MISSING)
        if before != after:
            changes.append(ConfigChange(key, before, after))
    return changes

def _ordered_metrics(names: Sequence[str]) -> List[str]:
    names = set(names)
    return [m for m in DISPLAY_ORDER if m in names] + sorted(names - set(DISPLAY_ORDER))



def diff_metrics(base: RunSnapshot, cand: RunSnapshot, metrics: Sequence[str]) -> List[MetricDelta]:
    """Per metric, over rows where BOTH runs have a value.

    System failures contribute 0.0; evaluator failures contribute nothing.
    """

    common = sorted(set(base.rows) & set(cand.rows))
    out: List[MetricDelta] = []

    for name in metrics:
        pairs = []
        for qa_id in common:
            x = base.rows[qa_id].values.get(name)
            y = cand.rows[qa_id].values.get(name)

            if x is not None and y is not None:
                pairs.append((x,y))

        better = sum(1 for x, y in pairs if not math.isclose(x, y, abs_tol=_TIE_TOL) and y > x)
        worse = sum(1 for x, y in pairs if not math.isclose(x, y, abs_tol=_TIE_TOL) and y < x)

        n = len(pairs)
        before = sum(x for x, _ in pairs) / n if n else None
        after = sum(y for _, y in pairs) / n if n else None

        out.append(
            MetricDelta(
                name=name,
                before=before,
                after=after,
                delta=(after - before) if n else None,
                n_common=n,
                better=better,
                worse=worse,
                tie=n - better - worse,
                applicable_base=name in base.applicable,
                applicable_cand=name in cand.applicable,
            )
        )

    return out


# ---------------------------------------------------------------- entry point

def compare_runs(repo: Repository, baseline_id: str, candidate_id: str) -> ComparisonResult:
    """Compare two stored runs"""

    base = load_snapshot(repo, baseline_id)
    cand = load_snapshot(repo, candidate_id)

    warnings, ungated = check_comparable(base, cand)

    names = _ordered_metrics(base.applicable | cand.applicable)

    only_b = sorted(set(base.rows) - set(cand.rows))
    only_c = sorted(set(cand.rows) - set(base.rows))

    return ComparisonResult(
        baseline=base,
        candidate=cand,
        config_changes=diff_config(base.run.config, cand.run.config),
        legacy_run_ids=_legacy_run_ids(base, cand),
        warnings=warnings,
        ungated=ungated,
        metrics=diff_metrics(base, cand, names),
        common_row_ids=tuple(sorted(set(base.rows) & set(cand.rows))),
        only_in_baseline=tuple(only_b),
        only_in_candidate=tuple(only_c),
    )