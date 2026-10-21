"""
tests/unit/test_diagnostics.py

ragregress/evaluation/diagnostics.py on synthetic data: worst-regressed rows and the
per-category / per-failure-mode breakdown. No database, no CLI, no LLM. Gates are injected.

Run (repo root):  python -m pytest tests/unit/test_diagnostics.py -v
"""

from __future__ import annotations
 
from types import SimpleNamespace
from typing import Iterable, Optional
 
import pytest
 
from ragregress.evaluation.comparator import (
    ComparisonResult,
    RowKind,
    RowView,
    RunSnapshot,
    diff_metrics,
)
from ragregress.evaluation.diagnostics import (
    NO_LABEL,
    SMALL_N,
    _row_note,
    diagnose,
    render_text,
)
from ragregress.evaluation.regression import Gate, assess_regression
 
GATES = {
    "score": Gate(0.05),
    "other": Gate(0.05),
    "latency": Gate(0.05, higher_is_better=False),
}
# "extra" has no gate -> must never appear in diagnostics.
 
 
# ---------------------------------------------------------------- builders
 
def _row(qa_id, category="c1", failure_mode="pass", kind=RowKind.MEASURED, **values) -> RowView:
    return RowView(qa_id, category, failure_mode, kind, dict(values))
 
 
def _sysfail(qa_id, *metrics, category="c1", failure_mode="failed") -> RowView:
    return RowView(qa_id, category, failure_mode, RowKind.SYSTEM_FAILURE, {m: 0.0 for m in metrics})
 
 
def _evalfail(qa_id, *metrics, category="c1", failure_mode="judge_failed") -> RowView:
    return RowView(qa_id, category, failure_mode, RowKind.EVALUATOR_FAILURE, {m: None for m in metrics})
 
 
def _snapshot(run_id: str, rows: Iterable[RowView]) -> RunSnapshot:
    rows = list(rows)
    applicable = frozenset(name for r in rows for name in r.values)
    return RunSnapshot(run=SimpleNamespace(run_id=run_id), rows={r.qa_id: r for r in rows}, applicable=applicable)
 
 
def _diagnose(base_rows, cand_rows, ungated=None, top_n=5):
    base, cand = _snapshot("base", base_rows), _snapshot("cand", cand_rows)
    names = sorted(base.applicable | cand.applicable)
    result = ComparisonResult(
        baseline=base,
        candidate=cand,
        config_changes=[],
        legacy_run_ids=(),
        warnings=[],
        ungated=dict(ungated or {}),
        metrics=diff_metrics(base, cand, names),
        common_row_ids=tuple(sorted(set(base.rows) & set(cand.rows))),
        only_in_baseline=tuple(sorted(set(base.rows) - set(cand.rows))),
        only_in_candidate=tuple(sorted(set(cand.rows) - set(base.rows))),
    )
    report = assess_regression(result, GATES)
    return diagnose(result, report, top_n=top_n), report
 
 
def _metric(d, name):
    return next(m for m in d.metrics if m.metric == name)
 
 
def _group(d, dimension, label):
    return next(g for g in d.groups if g.dimension == dimension and g.label == label)
 
 
# ---------------------------------------------------------------- Q22: ranking and top-N
 
def test_rows_are_ranked_worst_first_and_cut_at_top_n():
    base = [_row(f"q{i}", score=1.0) for i in range(6)]
    cand = [_row(f"q{i}", score=1.0 - 0.1 * i) for i in range(6)]      # q0 unchanged ... q5 drops by 0.5
    d, _ = _diagnose(base, cand, top_n=3)
 
    assert [r.qa_id for r in _metric(d, "score").rows] == ["q5", "q4", "q3"]
    assert _metric(d, "score").n_worse == 5                              # the cut hides rows, not counts
 
 
def test_ties_in_worsening_are_ordered_by_qa_id():
    base = [_row(i, score=1.0) for i in ("qb", "qa", "qc")]
    cand = [_row(i, score=0.5) for i in ("qb", "qa", "qc")]
    d, _ = _diagnose(base, cand)
    assert [r.qa_id for r in _metric(d, "score").rows] == ["qa", "qb", "qc"]
 
 
def test_only_rows_that_got_worse_are_listed_and_counts_add_up():
    base = [_row("q1", score=0.5), _row("q2", score=0.5), _row("q3", score=0.5)]
    cand = [_row("q1", score=0.2), _row("q2", score=0.9), _row("q3", score=0.5)]
    m = _metric(_diagnose(base, cand)[0], "score")
 
    assert [r.qa_id for r in m.rows] == ["q1"]
    assert (m.n_worse, m.n_better, m.n_tie, m.n_compared) == (1, 1, 1, 3)
 
 
def test_float_noise_is_a_tie_not_a_regression():
    base = [_row("q1", score=0.3)]
    cand = [_row("q1", score=0.1 + 0.2)]                                 # 0.30000000000000004
    m = _metric(_diagnose(base, cand)[0], "score")
    assert m.rows == [] and m.n_tie == 1
 
 
def test_row_fields_are_filled_in():
    base = [_row("q1", category="finance", failure_mode="pass", score=0.9)]
    cand = [_row("q1", category="finance", failure_mode="low_corr", score=0.4)]
    (r,) = _metric(_diagnose(base, cand)[0], "score").rows
 
    assert (r.qa_id, r.category, r.failure_mode) == ("q1", "finance", "low_corr")   # candidate's tag
    assert (r.before, r.after) == (0.9, 0.4)
    assert r.delta == pytest.approx(-0.5)
    assert r.worsening == pytest.approx(0.5)
 
 
def test_top_n_zero_hides_rows_but_keeps_counts():
    base = [_row("q1", score=1.0)]
    cand = [_row("q1", score=0.0)]
    d, _ = _diagnose(base, cand, top_n=0)
    assert _metric(d, "score").rows == []
    assert _metric(d, "score").n_worse == 1
    assert d.n_worse_any == 1
 
 
def test_lower_is_better_metric_worsens_when_the_value_goes_up():
    base = [_row("q1", latency=1.0), _row("q2", latency=1.0)]
    cand = [_row("q1", latency=1.5), _row("q2", latency=0.5)]
    m = _metric(_diagnose(base, cand)[0], "latency")
 
    assert m.higher_is_better is False
    assert [r.qa_id for r in m.rows] == ["q1"]
    assert m.rows[0].delta == pytest.approx(+0.5)          # signed like the delta table ...
    assert m.rows[0].worsening == pytest.approx(0.5)       # ... but worsening is always > 0 for "worse"
    assert (m.n_worse, m.n_better) == (1, 1)
 
 
# ---------------------------------------------------------------- rows worse on >= 1 gated metric
 
def test_row_worse_on_two_metrics_counts_once_in_worse_any():
    base = [_row("q1", score=1.0, other=1.0), _row("q2", score=1.0, other=1.0), _row("q3", score=1.0, other=1.0)]
    cand = [_row("q1", score=0.0, other=0.0), _row("q2", score=0.0, other=1.0), _row("q3", score=1.0, other=1.0)]
    d, _ = _diagnose(base, cand)
 
    assert d.n_common_rows == 3
    assert d.n_worse_any == 2
    assert _metric(d, "score").n_worse == 2 and _metric(d, "other").n_worse == 1
 
 
# ---------------------------------------------------------------- Q23: gated metrics only
 
def test_ungated_and_not_gated_metrics_are_not_diagnosed():
    base = [_row("q1", score=1.0, other=1.0, extra=1.0)]
    cand = [_row("q1", score=0.0, other=0.0, extra=0.0)]
    d, report = _diagnose(base, cand, ungated={"other": "judge_model differs"})
 
    assert d.gated == report.gated == ("score",)
    assert [m.metric for m in d.metrics] == ["score"]
    assert "other" not in d.groups[0].metrics and "extra" not in d.groups[0].metrics
 
 
def test_no_gated_metrics_means_nothing_to_diagnose():
    base = [_row("q1", extra=1.0)]
    cand = [_row("q1", extra=0.0)]
    d, _ = _diagnose(base, cand)
 
    assert d.gated == () and d.metrics == [] and d.groups == [] and d.n_worse_any == 0
    assert render_text(d) == ["worst regressions: no gated metrics, nothing to diagnose"]
 
 
# ---------------------------------------------------------------- Q24: which rows
 
def test_system_failure_counts_as_zero_and_is_tagged():
    base = [_row("q1", score=0.8), _row("q2", score=0.8)]
    cand = [_sysfail("q1", "score"), _row("q2", score=0.8)]
    (r,) = _metric(_diagnose(base, cand)[0], "score").rows
 
    assert (r.qa_id, r.after) == ("q1", 0.0)
    assert r.note == "system failure in candidate"
 
 
@pytest.mark.parametrize(
    ("base_kind", "cand_kind", "note"),
    [
        (RowKind.MEASURED, RowKind.MEASURED, ""),
        (RowKind.MEASURED, RowKind.SYSTEM_FAILURE, "system failure in candidate"),
        (RowKind.SYSTEM_FAILURE, RowKind.MEASURED, "system failure in baseline"),
        (RowKind.SYSTEM_FAILURE, RowKind.SYSTEM_FAILURE, "system failure in both runs"),
        (RowKind.MEASURED, RowKind.EVALUATOR_FAILURE, ""),
    ],
)
def test_row_note_for_each_combination_of_row_kinds(base_kind, cand_kind, note):
    assert _row_note(base_kind, cand_kind) == note
 
 
def test_system_failure_in_both_runs_is_a_tie_not_a_regression():
    d, _ = _diagnose([_sysfail("q1", "score")], [_sysfail("q1", "score")])
    assert _metric(d, "score").rows == [] and _metric(d, "score").n_tie == 1
 
 
def test_evaluator_failure_rows_are_excluded_from_everything():
    base = [_row("q1", score=1.0), _row("q2", score=1.0)]
    cand = [_evalfail("q1", "score"), _row("q2", score=1.0)]
    d, _ = _diagnose(base, cand)
 
    assert _metric(d, "score").n_compared == 1
    assert _metric(d, "score").rows == []
    assert d.n_common_rows == 2                            # the row is still a common row ...
    assert d.n_worse_any == 0                              # ... it just has nothing to compare
 
 
def test_only_common_rows_are_considered():
    base = [_row("q1", score=1.0), _row("only_base", score=1.0)]
    cand = [_row("q1", score=0.5), _row("only_cand", score=0.0)]
    d, _ = _diagnose(base, cand)
 
    assert d.n_common_rows == 1
    assert [r.qa_id for r in _metric(d, "score").rows] == ["q1"]
 
 
# ---------------------------------------------------------------- Q25: groups
 
def test_category_groups_count_rows_and_regressions():
    base = [_row(f"a{i}", category="finance", score=1.0) for i in range(4)] + \
           [_row(f"b{i}", category="sports", score=1.0) for i in range(2)]
    cand = [_row("a0", category="finance", score=0.0), _row("a1", category="finance", score=0.5),
            _row("a2", category="finance", score=1.0), _row("a3", category="finance", score=1.0),
            _row("b0", category="sports", score=1.0), _row("b1", category="sports", score=1.0)]
    d, _ = _diagnose(base, cand)
 
    fin = _group(d, "category", "finance")
    assert (fin.n, fin.n_worse_any, fin.small_n) == (4, 2, False)
    cell = fin.metrics["score"]
    assert cell.mean_delta == pytest.approx(-0.375)       # (-1.0 - 0.5 + 0 + 0) / 4
    assert (cell.worse, cell.better, cell.tie, cell.n) == (2, 0, 2, 4)
 
    sports = _group(d, "category", "sports")
    assert (sports.n, sports.n_worse_any, sports.small_n) == (2, 0, True)      # fewer than SMALL_N rows
 
 
def test_groups_are_sorted_by_regressed_rows_then_label():
    base = [_row("q1", category="zeta", score=1.0), _row("q2", category="alpha", score=1.0),
            _row("q3", category="mid", score=1.0)]
    cand = [_row("q1", category="zeta", score=0.0), _row("q2", category="alpha", score=1.0),
            _row("q3", category="mid", score=1.0)]
    d, _ = _diagnose(base, cand)
    labels = [g.label for g in d.groups if g.dimension == "category"]
    assert labels == ["zeta", "alpha", "mid"]
 
 
def test_failure_mode_groups_use_the_candidate_tag():
    base = [_row("q1", failure_mode="pass", score=1.0), _row("q2", failure_mode="pass", score=1.0)]
    cand = [_row("q1", failure_mode="low_corr", score=0.0), _row("q2", failure_mode="pass", score=1.0)]
    d, _ = _diagnose(base, cand)
 
    assert _group(d, "failure_mode", "low_corr").n_worse_any == 1
    assert _group(d, "failure_mode", "pass").n_worse_any == 0
    assert _group(d, "failure_mode", "pass").n == 1       # q1 is NOT counted under its baseline tag
 
 
def test_null_category_and_failure_mode_get_a_visible_label():
    base = [_row("q1", category=None, failure_mode=None, score=1.0)]
    cand = [_row("q1", category=None, failure_mode=None, score=0.0)]
    d, _ = _diagnose(base, cand)
    assert _group(d, "category", NO_LABEL).n == 1
    assert _group(d, "failure_mode", NO_LABEL).n == 1
 
 
def test_group_cell_is_none_when_nothing_was_comparable():
    base = [_row("q1", category="x", score=1.0)]
    cand = [_evalfail("q1", "score", category="x")]
    cell = _group(_diagnose(base, cand)[0], "category", "x").metrics["score"]
    assert cell.n == 0 and cell.mean_delta is None
 
 
def test_small_n_boundary_follows_the_constant():
    ids = [f"q{i}" for i in range(SMALL_N)]
    base = [_row(i, category="exact", score=1.0) for i in ids]
    cand = [_row(i, category="exact", score=1.0) for i in ids]
    assert _group(_diagnose(base, cand)[0], "category", "exact").small_n is False
    assert _group(_diagnose(base[:-1], cand[:-1])[0], "category", "exact").small_n is True
 
 
# ---------------------------------------------------------------- rendering
 
def test_render_text_lists_rows_groups_and_markers():
    base = [_row("q1", category="finance", score=1.0), _row("q2", category="sports", score=1.0)]
    cand = [_sysfail("q1", "score", category="finance"), _row("q2", category="sports", score=1.0)]
    text = "\n".join(render_text(_diagnose(base, cand)[0]))
 
    assert "1 of 2 common rows are worse on >= 1 gated metric" in text
    assert "score  (higher is better; 1 worse / 0 better / 1 tie of 2 compared)" in text
    assert "q1" in text and "[system failure in candidate]" in text
    assert "by category" in text and "by failure mode (candidate tag)" in text
    assert "finance*" in text                              # n = 1 < SMALL_N -> starred
    assert "-1.000 1/0" in text                            # group cell: mean delta, worse/better
 
 
def test_render_text_keeps_columns_apart_for_long_real_world_ids():
    # real qa_ids are ~24 hex characters, longer than any fixed column; they must not run into the category
    long_id, long_category = "5725bad5271a42140099d0bd", "simple_with_condition"
    base = [_row(long_id, category=long_category, failure_mode="low_correctness", score=1.0)]
    cand = [_row(long_id, category=long_category, failure_mode="low_correctness", score=0.2)]
    lines = render_text(_diagnose(base, cand)[0])
 
    (row_line,) = [ln for ln in lines if ln.strip().startswith(long_id)]
    assert row_line.split()[:3] == [long_id, long_category, "low_correctness"]
    assert f"{long_id}{long_category}" not in "\n".join(lines)
 
 
def test_render_text_says_when_nothing_got_worse():
    base = [_row("q1", score=0.5)]
    cand = [_row("q1", score=0.9)]
    text = "\n".join(render_text(_diagnose(base, cand)[0]))
    assert "(no row got worse)" in text
 
 
def test_render_text_respects_top_n():
    base = [_row(f"q{i}", score=1.0) for i in range(5)]
    cand = [_row(f"q{i}", score=0.0) for i in range(5)]
    text = "\n".join(render_text(_diagnose(base, cand, top_n=2)[0]))
    assert "q0" in text and "q1" in text and "q2" not in text
    assert "(top 2 per metric)" in text
 