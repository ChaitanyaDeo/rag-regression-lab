"""
ragregress/evaluation/diagnostics.py

"Which queries regressed, and where?" -- a read-only breakdown of one comparison.
Pure logic over a ComparisonResult + RegressionReport: no DB access, no printing side effects
(`render_text` only returns lines), and it never changes the verdict or the exit code.

"""

from __future__ import annotations
 
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
 
from ragregress.evaluation.comparator import ComparisonResult, RowKind
from ragregress.evaluation.regression import RegressionReport
 
TIE_TOL = 1e-9       # |worsening| <= TIE_TOL is a tie (float noise), same tolerance as the comparator
SMALL_N = 3          # groups with fewer rows than this are flagged: their means are anecdotes
NO_LABEL = "(none)"  # group label for a row whose category / failure_mode is NULL
 
DIMENSIONS = ("category", "failure_mode")
 
 
@dataclass(frozen=True)
class RowRegression:
    qa_id: str
    category: Optional[str]
    failure_mode: Optional[str]     # the candidate row's tag
    before: float
    after: float
    delta: float                    # after - before (signed, same convention as the delta table)
    worsening: float                # > 0 means worse, whatever the metric's direction
    note: str = ""                  # "system failure in candidate" etc.
 
 
@dataclass(frozen=True)
class MetricWorst:
    metric: str
    higher_is_better: bool
    n_compared: int                 # common rows measured in both runs
    n_worse: int
    n_better: int
    n_tie: int
    rows: List[RowRegression]       # top-N worst, worst first (only rows that actually got worse)
 
 
@dataclass(frozen=True)
class GroupMetric:
    n: int                          # rows in the group measured in both runs for this metric
    mean_delta: Optional[float]     # mean of (after - before); None when n == 0
    worse: int
    better: int
    tie: int
 
 
@dataclass(frozen=True)
class GroupStat:
    dimension: str                  # "category" | "failure_mode"
    label: str
    n: int                          # common rows in the group
    n_worse_any: int                # rows worse on >= 1 gated metric
    small_n: bool
    metrics: Dict[str, GroupMetric]
 
 
@dataclass(frozen=True)
class Diagnostics:
    gated: Tuple[str, ...]
    n_common_rows: int
    n_worse_any: int                # common rows worse on >= 1 gated metric
    top_n: int
    metrics: List[MetricWorst]
    groups: List[GroupStat]         # category groups first, then failure_mode groups
 
 
# ---------------------------------------------------------------- computation
 
def _worsening(before: float, after: float, higher_is_better: bool) -> float:
    return (before - after) if higher_is_better else (after - before)
 
 
def _row_note(base_kind: RowKind, cand_kind: RowKind) -> str:
    base_sys, cand_sys = base_kind is RowKind.SYSTEM_FAILURE, cand_kind is RowKind.SYSTEM_FAILURE
    if base_sys and cand_sys:
        return "system failure in both runs"
    if cand_sys:
        return "system failure in candidate"
    if base_sys:
        return "system failure in baseline"
    return ""
 
 
def diagnose(
    result: ComparisonResult,
    report: RegressionReport,
    top_n: int = 5,
    small_n: int = SMALL_N,
) -> Diagnostics:
    """Worst-regressed rows and per-group breakdown for the gated metrics of one comparison."""
    gated = tuple(report.gated)
    direction: Dict[str, bool] = {
        m.name: (True if m.higher_is_better is None else bool(m.higher_is_better))
        for m in report.metrics
        if m.name in gated
    }
 
    base_rows, cand_rows = result.baseline.rows, result.candidate.rows
    common = tuple(result.common_row_ids)
 
    # cells[metric][qa_id] = (before, after, worsening)   -- only rows measured in both runs
    cells: Dict[str, Dict[str, Tuple[float, float, float]]] = {name: {} for name in gated}
    worse_any: set = set()
    for qa_id in common:
        base_row, cand_row = base_rows[qa_id], cand_rows[qa_id]
        for name in gated:
            before, after = base_row.values.get(name), cand_row.values.get(name)
            if before is None or after is None:
                continue
            w = _worsening(before, after, direction[name])
            cells[name][qa_id] = (before, after, w)
            if w > TIE_TOL:
                worse_any.add(qa_id)
 
    # ---- per-metric worst rows
    metric_reports: List[MetricWorst] = []
    for name in gated:
        per_row = cells[name]
        worsened = sorted(
            ((qa_id, c) for qa_id, c in per_row.items() if c[2] > TIE_TOL),
            key=lambda item: (-item[1][2], item[0]),
        )
        rows = []
        for qa_id, (before, after, w) in worsened[: max(top_n, 0)]:
            rows.append(
                RowRegression(
                    qa_id=qa_id,
                    category=cand_rows[qa_id].category,
                    failure_mode=cand_rows[qa_id].failure_mode,
                    before=before,
                    after=after,
                    delta=after - before,
                    worsening=w,
                    note=_row_note(base_rows[qa_id].kind, cand_rows[qa_id].kind),
                )
            )
        n_worse = sum(1 for c in per_row.values() if c[2] > TIE_TOL)
        n_better = sum(1 for c in per_row.values() if c[2] < -TIE_TOL)
        metric_reports.append(
            MetricWorst(
                metric=name,
                higher_is_better=direction[name],
                n_compared=len(per_row),
                n_worse=n_worse,
                n_better=n_better,
                n_tie=len(per_row) - n_worse - n_better,
                rows=rows,
            )
        )
 
    # ---- groups (nothing to break down when no metric is gated)
    groups: List[GroupStat] = []
    for dimension in DIMENSIONS if gated else ():
        buckets: Dict[str, List[str]] = {}
        for qa_id in common:
            value = getattr(cand_rows[qa_id], dimension)
            buckets.setdefault(NO_LABEL if value is None else str(value), []).append(qa_id)
 
        dimension_groups = []
        for label, ids in buckets.items():
            per_metric: Dict[str, GroupMetric] = {}
            for name in gated:
                got = [cells[name][i] for i in ids if i in cells[name]]
                n = len(got)
                per_metric[name] = GroupMetric(
                    n=n,
                    mean_delta=(sum(a - b for b, a, _ in got) / n) if n else None,
                    worse=sum(1 for _, _, w in got if w > TIE_TOL),
                    better=sum(1 for _, _, w in got if w < -TIE_TOL),
                    tie=sum(1 for _, _, w in got if abs(w) <= TIE_TOL),
                )
            dimension_groups.append(
                GroupStat(
                    dimension=dimension,
                    label=label,
                    n=len(ids),
                    n_worse_any=sum(1 for i in ids if i in worse_any),
                    small_n=len(ids) < small_n,
                    metrics=per_metric,
                )
            )
        dimension_groups.sort(key=lambda g: (-g.n_worse_any, g.label))
        groups.extend(dimension_groups)
 
    return Diagnostics(
        gated=gated,
        n_common_rows=len(common),
        n_worse_any=len(worse_any),
        top_n=top_n,
        metrics=metric_reports,
        groups=groups,
    )
 
 
# ---------------------------------------------------------------- text rendering
 
def _cell(group_metric: Optional[GroupMetric]) -> str:
    if group_metric is None or group_metric.mean_delta is None:
        return "n/a"
    return f"{group_metric.mean_delta:+.3f} {group_metric.worse}/{group_metric.better}"
 
 
def render_text(d: Diagnostics) -> List[str]:
    """Plain-text lines for the CLI (and, later, anything that wants a text dump)."""
    if not d.gated:
        return ["worst regressions: no gated metrics, nothing to diagnose"]
 
    lines: List[str] = [
        f"worst regressions: {d.n_worse_any} of {d.n_common_rows} common rows are worse on "
        f">= 1 gated metric (top {d.top_n} per metric)"
    ]
 
    # Column widths follow the data: real qa_ids are ~24 characters and would otherwise run into the next column.
    all_rows = [r for m in d.metrics for r in m.rows]
    id_w = max([len("qa_id")] + [len(r.qa_id) for r in all_rows]) + 2
    cat_w = max([len("category")] + [len(r.category or NO_LABEL) for r in all_rows]) + 2
    mode_w = max([len("failure_mode")] + [len(r.failure_mode or NO_LABEL) for r in all_rows]) + 2
 
    for m in d.metrics:
        direction = "higher is better" if m.higher_is_better else "lower is better"
        lines.append("")
        lines.append(
            f"{m.metric}  ({direction}; {m.n_worse} worse / {m.n_better} better / {m.n_tie} tie "
            f"of {m.n_compared} compared)"
        )
        if not m.rows:
            lines.append("  (no row got worse)" if m.n_worse == 0 else "  (rows hidden: top_n = 0)")
            continue
        lines.append(
            f"  {'qa_id':<{id_w}}{'category':<{cat_w}}{'failure_mode':<{mode_w}}"
            f"{'before':>8}{'after':>8}{'delta':>9}"
        )
        for r in m.rows:
            suffix = f"  [{r.note}]" if r.note else ""
            lines.append(
                f"  {r.qa_id:<{id_w}}{(r.category or NO_LABEL):<{cat_w}}{(r.failure_mode or NO_LABEL):<{mode_w}}"
                f"{r.before:>8.3f}{r.after:>8.3f}{r.delta:>+9.3f}{suffix}"
            )
 
    names: Sequence[str] = d.gated
    for dimension, title in (("category", "by category"), ("failure_mode", "by failure mode (candidate tag)")):
        groups = [g for g in d.groups if g.dimension == dimension]
        if not groups:
            continue
        lines.append("")
        lines.append(f"{title}  (cells: mean delta  worse/better;  * = fewer than {SMALL_N} rows)")
        label_w = max([len("group")] + [len(g.label) + 1 for g in groups]) + 2
        header = f"  {'group':<{label_w}}{'n':>4}{'worse':>7}  " + "".join(f"{name:<18}" for name in names)
        lines.append(header)
        for g in groups:
            label = g.label + ("*" if g.small_n else "")
            lines.append(
                f"  {label:<{label_w}}{g.n:>4}{g.n_worse_any:>7}  "
                + "".join(f"{_cell(g.metrics.get(name)):<18}" for name in names)
            )
 
    return lines