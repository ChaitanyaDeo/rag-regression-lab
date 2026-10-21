"""
ragregress/evaluation/regression.py

Turns a ComparisonResult (comparator.py) into a verdict.
Pure logic: no DB access, no printing, no CLI

"""
from __future__ import annotations
 
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
 
from ragregress.evaluation.comparator import ComparisonResult, RowView, RunSnapshot

VALIDITY_MIN = 0.80     # fraction of attempted rows that must be measured, per run
BOUNDARY_EPS = 1e-9     # float tolerance on "strictly greater than the threshold"

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


# Process exit codes for `ragregress compare`: 8 = invalid input is an ERROR, not a verdict
# (ComparisonInputError in core/errors.py). Kept here, next to Verdict, so the mapping has one place

VERDICT_EXIT_CODES: Dict[Verdict, int] = {
    Verdict.PASS: 0,
    Verdict.FAIL: 6,
    Verdict.INCONCLUSIVE: 7,
}

class MetricStatus(str, Enum):
    OK = "ok"                          # gated, within threshold
    REGRESSED = "regressed"            # gated, worsened by more than the threshold
    NOT_GATED = "not gated"            # no declared threshold / unknown to the registry
    UNGATED = "ungated"                # has a threshold, disabled for this pair (see note)
    NOT_APPLICABLE = "n/a"             # metric absent from one of the runs
    NO_DATA = "no data"                # gated, but no common measured rows

_GATED_STATUSES = frozenset({MetricStatus.OK, MetricStatus.REGRESSED, MetricStatus.NO_DATA})

@dataclass
class Gate:
    threshold: float         # largest tolerated worsening, in the metric's own units
    higher_is_better: bool = True

@dataclass(frozen=True)
class Validity:
    measured: int
    attempted: int

    @property
    def ratio(self) -> float:
        return self.measured / self.attempted if self.attempted else 0.0

    def is_valid(self, minimum: float = VALIDITY_MIN) -> bool:
        return self.attempted > 0 and self.measured + BOUNDARY_EPS >= minimum * self.attempted


@dataclass(frozen=True)
class MetricReport:
    name:str
    status: MetricStatus
    before: Optional[float]
    after: Optional[float]
    delta: Optional[float]
    n_common: int
    better: int
    worse: int
    tie: int
    threshold: Optional[float]      # None when the metric has no gate
    higher_is_better: Optional[bool]
    note: str = ""                  # reason for UNGATED / NOT_APPLICABLE / NOT_GATED / NO_DATA


@dataclass(frozen=True)
class RegressionReport:
    verdict: Verdict
    reasons: List[str]
    metrics: List[MetricReport]
    gated: Tuple[str, ...]
    validity_baseline: Validity
    validity_candidate: Validity
 
    @property
    def exit_code(self) -> int:
        return VERDICT_EXIT_CODES[self.verdict]

# ---------------------------------------------------------------- gates

def gates_from_registry(registry: Optional[Mapping[str, type]] = None) -> Dict[str, Gate]:
    """Read each metric CLASS's declared gate. Metrics with `regression_threshold=None` are omitted.

    `registry` defaults to METRIC_REGISTRY; imported lazily so this module (and its tests) do not
    pull in the metric implementations (LLM judge, etc.) unless the default is actually used.
    """

    if registry is None:
        from ragregress.evaluation.metrics import METRIC_REGISTRY as registry

    gates: Dict[str, Gate] = {}

    for name, cls in registry.items():
        threshold = getattr(cls, "regression_threshold", None)
        if threshold is not None:
            gates[name] = Gate(float(threshold), bool(getattr(cls, "higher_is_better", True)))
    return gates

# ---------------------------------------------------------------- validity

def _is_measured(row: RowView, gated: Sequence[str])-> bool:
    return all(row.values.get(name) is not None for name in gated)

def run_validity(snapshot: RunSnapshot, gated: Sequence[str]) -> Validity:
    """Rows with a value for every gated metric / all attempted rows 
    
    System failures carry 0.0 on every applicable metric -> measured.
    Evaluator failures carry None -> unmeasured.
    """

    measured = sum(1 for row in snapshot.rows.values() if _is_measured(row, gated))

    return Validity(measured, snapshot.attempted)

# ------------------------------------------------------------------- per-metric status

def _worsening(before: float, after: float, gate: Gate) -> float:
    return (before - after) if gate.higher_is_better else (after - before)


def _classify(delta, gate: Optional[Gate], ungated: Mapping[str, str]) -> MetricReport:
    name = delta.name
    base = dict(
        name=name,
        before=delta.before,
        after=delta.after,
        delta=delta.delta,
        n_common=delta.n_common,
        better=delta.better,
        worse=delta.worse,
        tie=delta.tie,
        threshold=gate.threshold if gate else None,
        higher_is_better=gate.higher_is_better if gate else None,
    )

    if not (delta.applicable_base and delta.applicable_cand):
        missing = "baseline" if not delta.applicable_base else "candidate"

        return MetricReport(status=MetricStatus.NOT_APPLICABLE, note=f"absent in {missing}", **base)

    if gate is None:
        return MetricReport(status=MetricStatus.NOT_GATED, note="no declared regression threshold", **base)

    if name in ungated:
        return MetricReport(status=MetricStatus.UNGATED, note=ungated[name], **base)

    if delta.n_common==0 or delta.before is None or delta.after is None:
        return MetricReport(status=MetricStatus.NO_DATA, note="no rows measured in both runs", **base)

    regressed = _worsening(delta.before, delta.after, gate) > gate.threshold + BOUNDARY_EPS

    return MetricReport(status=MetricStatus.REGRESSED if regressed else MetricStatus.OK, **base)

# ---------------------------------------------------------------- verdict

def assess_regression(
    result: ComparisonResult,
    gates: Optional[Mapping[str, Gate]] = None,
    validity_min: float = VALIDITY_MIN,
) -> RegressionReport:

    """Verdict for one comparison. `gates=None` reads them from METRIC_REGISTRY"""

    if gates is None:
        gates = gates_from_registry()

    reports = [_classify(d, gates.get(d.name), result.ungated) for d in result.metrics]

    gated = tuple(r.name for r in reports if r.status in _GATED_STATUSES)

    validity_base = run_validity(result.baseline, gated)
    validity_cand = run_validity(result.candidate, gated)

    inconclusive: List[str] = []

    if not gated:
        inconclusive.append("no gated metrics in common")

    for label, validity in (("baseline", validity_base), ("candidate", validity_cand)):
        if gated and not validity.is_valid(validity_min):
            inconclusive.append(
                f"{label} validity {validity.measured}/{validity.attempted} "
                f"= {validity.ratio:.0%} < {validity_min:.0%}"
            ) 

    for r in reports:
        if r.status is MetricStatus.NO_DATA:
            inconclusive.append(f"{r.name}: {r.note}")

    regressed = [r for r in reports if r.status is MetricStatus.REGRESSED]

    regressed_text = [
        f"{r.name} worsened by "
        f"{_worsening(r.before, r.after, Gate(r.threshold, bool(r.higher_is_better))):.4f} "
        f"(threshold {r.threshold:.4f})"
        for r in regressed
    ]

    if inconclusive:
        verdict = Verdict.INCONCLUSIVE
        reasons = inconclusive
        if regressed_text:
            reasons = reasons + ["also flagged regressed (not trusted): " + "; ".join(regressed_text)]
    elif regressed:
        verdict = Verdict.FAIL
        reasons = regressed_text
    else:
        verdict = Verdict.PASS
        reasons = [f"no gated metric worsened beyond its threshold ({', '.join(gated)})"]
 
    return RegressionReport(
        verdict=verdict,
        reasons=reasons,
        metrics=reports,
        gated=gated,
        validity_baseline=validity_base,
        validity_candidate=validity_cand,
    )