"""
ragregress/report.py

HTML report for one comparison (baseline vs candidate).

- build_report_context(...) - pure: comparison objects + already-loaded data -> ReportContext
                             (plain strings and css class names; no DB, no files)
- generate_report(...) - I/O: reads the repository, passages.json and the split file,
                         then calls build_report_context

- render_html / write_report Jinja2 (autoescape, StrictUndefined) -> self-contained HTML

The report never changes the verdict: it renders what compare_runs / assess_regression /
diagnose decided. Invalid input (ComparisonInputError, exit 8) propagates to the caller.
Missing passages / split files degrade to ids-only with a warning 
"""

from __future__ import annotations
 
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ragregress.core.errors import ComparisonInputError
from ragregress.evaluation.comparator import MISSING, ComparisonResult, RowView, compare_runs
from ragregress.evaluation.diagnostics import NO_LABEL, Diagnostics, diagnose

from ragregress.evaluation.regression import (
    Gate,
    MetricStatus,
    RegressionReport,
    Verdict,
    assess_regression,
)
from ragregress.storage.models import QueryResult
from ragregress.storage.repositories import Repository


TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "index.html"
 
DEFAULT_PASSAGES = Path("data") / "passages.json"
DEFAULT_SPLITS_DIR = Path("data") / "splits"
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_TOP = 5

LOG_LINK = "logs/{run_id}.jsonl#{qa_id}"        # Stub target
LOG_NOTE = "raw log lines available Day 13"

DASH = "—"
HASH_SHORT = 12
TIE_TOL = 1e-9
TARGET_EPS = 1e-9

NO_TARGETS_NOTE = "No targets configured for this comparison (pass --targets PATH to add them)."

LATENCY_STAGES = ("total", "retrieve", "generate", "judge")   # display order;


VERDICT_CSS = {
    Verdict.PASS: "verdict-pass",
    Verdict.FAIL: "verdict-fail",
    Verdict.INCONCLUSIVE: "verdict-inconclusive",
}
VERDICT_HEADLINE = {
    Verdict.PASS: "PASS: no gated metric regressed beyond its threshold",
    Verdict.FAIL: "FAIL: at least one gated metric regressed beyond its threshold",
    Verdict.INCONCLUSIVE: "INCONCLUSIVE: this comparison cannot be trusted; do not read it as a pass",
}
STATUS_CSS = {
    MetricStatus.OK: "status-ok",
    MetricStatus.REGRESSED: "status-regressed",
    MetricStatus.NO_DATA: "status-warn",
    MetricStatus.UNGATED: "status-muted",
    MetricStatus.NOT_GATED: "status-muted",
    MetricStatus.NOT_APPLICABLE: "status-muted",
}

class TargetsFileError(ComparisonInputError):
    """The --targets file is missing, unparseable or malformed (exit 8, like other invalid input)."""
 
    def __init__(self, path, problem: str):
        self.path = str(path)
        self.problem = problem
        super().__init__(f"targets file {self.path}: {problem}")


@dataclass(frozen=True)
class TargetRange:
    label: str
    low: float
    high: float
 
 
@dataclass(frozen=True)
class TargetSpec:
    source: str                         # free text, e.g. "README.md"
    path: str
    ranges: Dict[str, TargetRange]      # metric name -> range, in file order

# ---------------------------------------------------------------- view model
 
@dataclass(frozen=True)
class VerdictView:
    label: str
    css: str
    headline: str
    reasons: List[str]

@dataclass(frozen=True)
class RunView:
    role: str
    run_id: str
    created_at: str
    split: str
    config_hash: str
    config_hash_short: str
    top_k: str
    validity: str          # "48/50 (96%)"
    legacy: bool

@dataclass(frozen=True)
class ConfigChangeRow:
    key: str
    before: str
    after: str
 
 
@dataclass(frozen=True)
class MetricRow:
    name: str
    before: str
    after: str
    delta: str
    delta_css: str
    n: int
    better: int
    worse: int
    tie: int
    threshold: str
    status: str
    status_css: str
    note: str

@dataclass(frozen=True)
class TargetRow:
    metric: str
    label: str
    target: str
    baseline: str
    baseline_status: str
    baseline_css: str
    candidate: str
    candidate_status: str
    candidate_css: str

@dataclass(frozen=True)
class LatencyRow:
    stage: str
    base_p50: str
    cand_p50: str
    delta_p50: str
    delta_p50_css: str
    base_p95: str
    cand_p95: str
    delta_p95: str
    delta_p95_css: str


@dataclass(frozen=True)
class GroupRow:
    label: str
    small_n: bool
    n: int
    n_worse: int
    cells: List[str]
 
 
@dataclass(frozen=True)
class GroupTable:
    title: str
    columns: List[str]
    rows: List[GroupRow]
 
 
@dataclass(frozen=True)
class PassageView:
    passage_id: str
    title: str
    text: str
    is_gold: bool
    found: bool            # False when passages.json is missing or lacks this id
    hits: int              # how many retrieved chunks came from this passage

@dataclass(frozen=True)
class SideView:
    role: str
    run_id: str
    row_kind: str
    query_status: str
    failure_mode: str
    answer: str
    metrics: List[Tuple[str, str]]
    faithfulness_reason: str
    last_error: str
    passages: List[PassageView]
    gold_not_retrieved: List[str]
    log_link: str
 
 
@dataclass(frozen=True)
class DrillDown:
    rank: int
    qa_id: str
    category: str
    question: str
    gold_answers: List[str]
    worst_metric: str
    worst_worsening: str
    n_metrics_worse: int
    baseline: SideView
    candidate: SideView

@dataclass(frozen=True)
class ReportContext:
    title: str
    generated_at: str
    verdict: VerdictView
    baseline: RunView
    candidate: RunView
    warnings: List[str]            # comparator warnings (top_k / judge mismatch, question sets ...)
    data_warnings: List[str]       # report-side degradations (missing passages / split ...)
    config_changes: List[ConfigChangeRow]
    metrics: List[MetricRow]
    targets: List[TargetRow]            # empty when no targets file was given
    target_note: str
    latency: List[LatencyRow]
    worst_summary: str
    groups: List[GroupTable]
    drilldowns: List[DrillDown]
    top: int
    log_note: str
 
 
@dataclass(frozen=True)
class _Pick:
    qa_id: str
    worst_metric: str
    max_worsening: float
    n_worse: int

# ---------------------------------------------------------------- formatting

def fmt(value: Optional[float], dp: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return DASH
    return f"{value:.{dp}f}"
 
 
def fmt_signed(value: Optional[float], dp: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return DASH
    return f"{value:+.{dp}f}"
 
 
def fmt_ms(value: Optional[float], signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return DASH
    return f"{value:+,.0f} ms" if signed else f"{value:,.0f} ms"
 
 
def fmt_config_value(value) -> str:
    if value is MISSING:
        return DASH
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def delta_css(delta: Optional[float], higher_is_better: Optional[bool]) -> str:
    if delta is None or higher_is_better is None:
        return "delta-neutral"
    if abs(delta) <= TIE_TOL:
        return "delta-tie"
    better = delta > 0 if higher_is_better else delta < 0
    return "delta-better" if better else "delta-worse"


def target_status(value: Optional[float], low: float, high: float) -> Tuple[str, str]:
    """(label, css) -- below / within / above the inclusive [low, high] range."""
    if value is None:
        return "n/a", "target-na"
    if value < low - TARGET_EPS:
        return "below", "target-below"
    if value > high + TARGET_EPS:
        return "above", "target-above"
    return "within", "target-within"

def default_output_path(baseline_id: str, candidate_id: str, reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path:
    return Path(reports_dir) / f"{baseline_id}__vs__{candidate_id}.html"

# ---------------------------------------------------------------- section builders

def _run_view(role: str, snapshot, validity) -> RunView:
    run = snapshot.run
    cfg = run.config or {}
    return RunView(
        role=role,
        run_id=run.run_id,
        created_at=str(run.created_at),
        split=str(cfg.get("eval_split") or DASH),
        config_hash=run.config_hash or DASH,
        config_hash_short=(run.config_hash or DASH)[:HASH_SHORT],
        top_k=str(run.k),
        validity=f"{validity.measured}/{validity.attempted} ({validity.ratio:.0%})",
        legacy="pipeline" not in cfg,
    )

def _metric_rows(report: RegressionReport) -> List[MetricRow]:
    rows = []
    for m in report.metrics:
        rows.append(
            MetricRow(
                name=m.name,
                before=fmt(m.before),
                after=fmt(m.after),
                delta=fmt_signed(m.delta),
                delta_css=delta_css(m.delta, m.higher_is_better),
                n=m.n_common,
                better=m.better,
                worse=m.worse,
                tie=m.tie,
                threshold=fmt(m.threshold),
                status=m.status.value,
                status_css=STATUS_CSS[m.status],
                note=m.note,
            )
        )
    return rows

def _target_rows(report: RegressionReport, spec: TargetSpec) -> Tuple[List[TargetRow], List[str]]:
    """Actuals = the same common-row means as the delta table, so each metric has one number on the page.
 
    A target for a metric neither run has is skipped with a warning (wrong targets file, or a typo).
    """
    by_name = {m.name: m for m in report.metrics}
    rows: List[TargetRow] = []
    warnings: List[str] = []
    for name, rng in spec.ranges.items():
        m = by_name.get(name)
        if m is None:
            warnings.append(f"targets file names metric {name!r}, which neither run has; skipped")
            continue
        label, low, high = rng.label, rng.low, rng.high
        before, after = m.before, m.after
        b_status, b_css = target_status(before, low, high)
        c_status, c_css = target_status(after, low, high)
        rows.append(
            TargetRow(
                metric=name,
                label=label,
                target=f"{low:.2f}–{high:.2f}",
                baseline=fmt(before),
                baseline_status=b_status,
                baseline_css=b_css,
                candidate=fmt(after),
                candidate_status=c_status,
                candidate_css=c_css,
            )
        )
    return rows, warnings

def _latency_rows(base: Mapping[str, Optional[float]], cand: Mapping[str, Optional[float]]) -> List[LatencyRow]:
    stages = set()
    for key in list(base) + list(cand):
        for prefix in ("latency_p50_", "latency_p95_"):
            if key.startswith(prefix):
                stages.add(key[len(prefix):])
    ordered = [s for s in LATENCY_STAGES if s in stages] + sorted(stages - set(LATENCY_STAGES))
 
    def diff(a, b):
        return (b - a) if a is not None and b is not None else None
 
    rows = []
    for stage in ordered:
        b50, c50 = base.get(f"latency_p50_{stage}"), cand.get(f"latency_p50_{stage}")
        b95, c95 = base.get(f"latency_p95_{stage}"), cand.get(f"latency_p95_{stage}")
        d50, d95 = diff(b50, c50), diff(b95, c95)
        rows.append(
            LatencyRow(
                stage=stage,
                base_p50=fmt_ms(b50), cand_p50=fmt_ms(c50),
                delta_p50=fmt_ms(d50, signed=True), delta_p50_css=delta_css(d50, higher_is_better=False),
                base_p95=fmt_ms(b95), cand_p95=fmt_ms(c95),
                delta_p95=fmt_ms(d95, signed=True), delta_p95_css=delta_css(d95, higher_is_better=False),
            )
        )
    return rows


_GROUP_TITLES = {
    "category": "By category",
    "failure_mode": "By failure mode (candidate tag; partly circular: tags derive from the same metrics)",
}
 
 
def _group_tables(diag: Diagnostics) -> List[GroupTable]:
    tables = []
    for dimension in ("category", "failure_mode"):
        groups = [g for g in diag.groups if g.dimension == dimension]
        if not groups:
            continue
        rows = []
        for g in groups:
            cells = []
            for name in diag.gated:
                gm = g.metrics.get(name)
                if gm is None or gm.mean_delta is None:
                    cells.append(DASH)
                else:
                    cells.append(f"{gm.mean_delta:+.3f} ({gm.worse} worse / {gm.better} better)")
            rows.append(GroupRow(label=g.label, small_n=g.small_n, n=g.n, n_worse=g.n_worse_any, cells=cells))
        tables.append(GroupTable(title=_GROUP_TITLES[dimension], columns=list(diag.gated), rows=rows))
    return tables


# ---------------------------------------------------------------- drill-down

def select_drilldowns(
    result: ComparisonResult, report: RegressionReport, diag: Diagnostics, top: int
) -> List[_Pick]:
    """Top rows by max worsening over gated metrics; tie -> more metrics worse; then qa_id (D9-Q9).
 
    Candidates come from the union of diag.metrics[*].rows. That union contains the overall top `top`
    as long as diag was built with top_n >= top: a row whose worst metric is m and that ranks in the
    overall top `top` has fewer than `top` rows above it on m, so it is in m's per-metric list.
    """
    gated = tuple(report.gated)
    direction = {
        m.name: (True if m.higher_is_better is None else bool(m.higher_is_better))
        for m in report.metrics if m.name in gated
    }
    candidates = {r.qa_id for m in diag.metrics for r in m.rows}
 
    picks: List[_Pick] = []
    for qa_id in candidates:
        b_row, c_row = result.baseline.rows[qa_id], result.candidate.rows[qa_id]
        worse: Dict[str, float] = {}
        for name in gated:
            x, y = b_row.values.get(name), c_row.values.get(name)
            if x is None or y is None:
                continue
            w = (x - y) if direction[name] else (y - x)
            if w > TIE_TOL:
                worse[name] = w
        if not worse:
            continue
        worst = min(worse, key=lambda n: (-worse[n], n))
        picks.append(_Pick(qa_id, worst, worse[worst], len(worse)))
 
    picks.sort(key=lambda p: (-p.max_worsening, -p.n_worse, p.qa_id))
    return picks[: max(top, 0)]
 
 
def _passage_views(
    qr: QueryResult, passages: Optional[Mapping[str, Tuple[str, str]]]
) -> Tuple[List[PassageView], List[str]]:
    hits: Dict[str, int] = {}
    order: List[str] = []
    for pid in qr.retrieved_ids or []:
        if pid not in hits:
            order.append(pid)
        hits[pid] = hits.get(pid, 0) + 1
    gold = list(qr.gold_ids or [])
    views = []
    for pid in order:
        entry = passages.get(pid) if passages is not None else None
        views.append(
            PassageView(
                passage_id=pid,
                title=entry[0] if entry else "",
                text=entry[1] if entry else "",
                is_gold=pid in gold,
                found=entry is not None,
                hits=hits[pid],
            )
        )
    return views, [g for g in gold if g not in hits]
 
 
def _faithfulness_reason(qr: QueryResult) -> str:
    entry = (qr.metric_results or {}).get("faithfulness")
    if isinstance(entry, dict) and entry.get("reason"):
        return str(entry["reason"])
    return ""
 
 
def _side(
    role: str,
    run_id: str,
    row: RowView,
    qr: Optional[QueryResult],
    gated: Sequence[str],
    passages: Optional[Mapping[str, Tuple[str, str]]],
) -> SideView:
    metrics = [(name, fmt(row.values.get(name))) for name in gated]
    if qr is None:
        return SideView(role, run_id, row.kind.value, DASH, row.failure_mode or NO_LABEL, DASH,
                        metrics, "", "", [], [], LOG_LINK.format(run_id=run_id, qa_id=row.qa_id))
    views, gold_missing = _passage_views(qr, passages)
    return SideView(
        role=role,
        run_id=run_id,
        row_kind=row.kind.value,
        query_status=qr.query_status or DASH,
        failure_mode=qr.failure_mode or NO_LABEL,
        answer=qr.generation_answer if qr.generation_answer is not None else DASH,
        metrics=metrics,
        faithfulness_reason=_faithfulness_reason(qr),
        last_error=qr.last_error or "",
        passages=views,
        gold_not_retrieved=gold_missing,
        log_link=LOG_LINK.format(run_id=run_id, qa_id=row.qa_id),
    )
 
 
# ---------------------------------------------------------------- pure entry point
 
def build_report_context(
    result: ComparisonResult,
    report: RegressionReport,
    diag: Diagnostics,
    *,
    base_results: Mapping[str, QueryResult],
    cand_results: Mapping[str, QueryResult],
    base_metrics: Mapping[str, Optional[float]],
    cand_metrics: Mapping[str, Optional[float]],
    passages: Optional[Mapping[str, Tuple[str, str]]],
    gold_answers: Optional[Mapping[str, List[str]]],
    targets: Optional[TargetSpec] = None,
    top: int = DEFAULT_TOP,
    data_warnings: Sequence[str] = (),
    generated_at: str = "",
) -> ReportContext:
    base_id, cand_id = result.baseline.run.run_id, result.candidate.run.run_id
    gated = tuple(report.gated)
    warnings_out = list(data_warnings)
 
    drilldowns: List[DrillDown] = []
    unknown_passages: set = set()
    for rank, pick in enumerate(select_drilldowns(result, report, diag, top), start=1):
        b_row, c_row = result.baseline.rows[pick.qa_id], result.candidate.rows[pick.qa_id]
        b_qr, c_qr = base_results.get(pick.qa_id), cand_results.get(pick.qa_id)
        b_side = _side("baseline", base_id, b_row, b_qr, gated, passages)
        c_side = _side("candidate", cand_id, c_row, c_qr, gated, passages)
        if passages is not None:
            unknown_passages.update(p.passage_id for s in (b_side, c_side) for p in s.passages if not p.found)
        question = (c_qr.question if c_qr else None) or (b_qr.question if b_qr else None) or DASH
        drilldowns.append(
            DrillDown(
                rank=rank,
                qa_id=pick.qa_id,
                category=c_row.category or NO_LABEL,
                question=question,
                gold_answers=list((gold_answers or {}).get(pick.qa_id, [])),
                worst_metric=pick.worst_metric,
                worst_worsening=fmt(pick.max_worsening),
                n_metrics_worse=pick.n_worse,
                baseline=b_side,
                candidate=c_side,
            )
        )
    if unknown_passages:
        warnings_out.append(
            f"{len(unknown_passages)} retrieved passage id(s) not found in passages.json: "
            + ", ".join(sorted(unknown_passages))
        )
 
    if targets is None:
        target_rows, target_note = [], NO_TARGETS_NOTE
    else:
        target_rows, target_warnings = _target_rows(report, targets)
        warnings_out.extend(target_warnings)
        target_note = (
            f"Targets: {targets.source or targets.path}. "
            "Actuals: means over the common rows, same as the metric table."
        )
 
    if gated:
        worst_summary = (
            f"{diag.n_worse_any} of {diag.n_common_rows} common rows are worse on at least one gated metric"
        )
    else:
        worst_summary = "no gated metrics: nothing to break down"
 
    return ReportContext(
        title=f"{base_id} vs {cand_id}",
        generated_at=generated_at,
        verdict=VerdictView(
            label=report.verdict.value,
            css=VERDICT_CSS[report.verdict],
            headline=VERDICT_HEADLINE[report.verdict],
            reasons=list(report.reasons),
        ),
        baseline=_run_view("baseline", result.baseline, report.validity_baseline),
        candidate=_run_view("candidate", result.candidate, report.validity_candidate),
        warnings=list(result.warnings),
        data_warnings=warnings_out,
        config_changes=[
            ConfigChangeRow(c.key, fmt_config_value(c.before), fmt_config_value(c.after))
            for c in result.config_changes
        ],
        metrics=_metric_rows(report),
        targets=target_rows,
        target_note=target_note,
        latency=_latency_rows(base_metrics, cand_metrics),
        worst_summary=worst_summary,
        groups=_group_tables(diag),
        drilldowns=drilldowns,
        top=top,
        log_note=LOG_NOTE,
    )
 
 
# ---------------------------------------------------------------- I/O
 
def _number(value, where: str, path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetsFileError(path, f"{where} must be a number, got {value!r}")
    return float(value)
 
 
def load_targets(path: Path) -> TargetSpec:
    """Read a targets YAML file. Any problem raises TargetsFileError (exit 8): the user asked for this file.
 
    Shape:
        source: README.md DEC-03 (2026-09-10)      # optional
        targets:
          recall_at_k: {label: "Recall@5", low: 0.80, high: 0.85}
          mrr: {low: 0.65, high: 0.75}             # label defaults to the metric name
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TargetsFileError(path, "not found") from None
    except (OSError, yaml.YAMLError) as exc:
        raise TargetsFileError(path, f"unreadable ({exc.__class__.__name__}: {exc})") from None
 
    if not isinstance(raw, dict) or not isinstance(raw.get("targets"), dict) or not raw["targets"]:
        raise TargetsFileError(path, "expected a non-empty 'targets:' mapping of metric -> {low, high}")
 
    ranges: Dict[str, TargetRange] = {}
    for name, entry in raw["targets"].items():
        if not isinstance(entry, dict):
            raise TargetsFileError(path, f"targets.{name} must be a mapping with low / high")
        missing = [k for k in ("low", "high") if k not in entry]
        if missing:
            raise TargetsFileError(path, f"targets.{name} is missing {', '.join(missing)}")
        low = _number(entry["low"], f"targets.{name}.low", path)
        high = _number(entry["high"], f"targets.{name}.high", path)
        if low > high:
            raise TargetsFileError(path, f"targets.{name}: low {low} > high {high}")
        label = entry.get("label", name)
        ranges[str(name)] = TargetRange(label=str(label), low=low, high=high)
 
    source = raw.get("source")
    return TargetSpec(source=str(source) if source else "", path=str(path), ranges=ranges)
 
 
def load_passages(path: Path) -> Tuple[Optional[Dict[str, Tuple[str, str]]], List[str]]:
    """passage_id -> (title, text). Returns (None, [warning]) instead of raising."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {p["passage_id"]: (p.get("title") or "", p.get("text") or "") for p in raw}, []
    except FileNotFoundError:
        return None, [f"passages file not found ({path}); drill-downs show passage ids only"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f"passages file unreadable ({path}: {exc!r}); drill-downs show passage ids only"]
 
 
def load_gold_answers(splits_dir: Path, split: Optional[str]) -> Tuple[Optional[Dict[str, List[str]]], List[str]]:
    """qa_id -> list of gold answer strings from data/splits/<split>.json."""
    if not split:
        return None, ["eval_split unknown (legacy run); gold answers not shown"]
    path = Path(splits_dir) / f"{split}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, List[str]] = {}
        for qa in raw:
            answer = qa.get("answer")
            out[qa["qa_id"]] = [str(a) for a in answer] if isinstance(answer, list) else (
                [str(answer)] if answer is not None else []
            )
        return out, []
    except FileNotFoundError:
        return None, [f"split file not found ({path}); gold answers not shown"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, [f"split file unreadable ({path}: {exc!r}); gold answers not shown"]
 
 
def generate_report(
    repo: Repository,
    baseline_id: str,
    candidate_id: str,
    *,
    passages_path: Path = DEFAULT_PASSAGES,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    top: int = DEFAULT_TOP,
    targets_path: Optional[Path] = None,
    gates: Optional[Mapping[str, Gate]] = None,
    now: Optional[datetime] = None,
) -> ReportContext:
    """Compare two stored runs and build the report context.
 
    ComparisonInputError (incl. TargetsFileError) propagates (exit 8). The targets file is read first,
    so a bad --targets fails before any database work.
    """
    targets = load_targets(targets_path) if targets_path is not None else None
    result = compare_runs(repo, baseline_id, candidate_id)
    report = assess_regression(result, gates)
    diag = diagnose(result, report, top_n=max(top, 0))     # top_n >= top: see select_drilldowns
 
    passages, warnings = load_passages(passages_path)
    split = (result.candidate.run.config or {}).get("eval_split") or (result.baseline.run.config or {}).get("eval_split")
    gold, split_warnings = load_gold_answers(splits_dir, split)
    warnings += split_warnings
 
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
 
    return build_report_context(
        result,
        report,
        diag,
        base_results={r.qa_id: r for r in repo.get_query_results(baseline_id)},
        cand_results={r.qa_id: r for r in repo.get_query_results(candidate_id)},
        base_metrics={m.metric_name: m.value for m in repo.get_metrics(baseline_id)},
        cand_metrics={m.metric_name: m.value for m in repo.get_metrics(candidate_id)},
        passages=passages,
        gold_answers=gold,
        targets=targets,
        top=top,
        data_warnings=warnings,
        generated_at=stamp,
    )
 
 
def render_html(ctx: ReportContext, template_dir: Optional[Path] = None) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_dir or TEMPLATE_DIR)),
        autoescape=True,                 # answers and passages are untrusted text
        undefined=StrictUndefined,       # a misspelled field fails loudly instead of rendering blank
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(TEMPLATE_NAME).render(ctx=ctx)
 
 
def write_report(ctx: ReportContext, out_path: Path, template_dir: Optional[Path] = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(ctx, template_dir), encoding="utf-8")
    return out_path
 