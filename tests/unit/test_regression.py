"""
tests/unit/test_regression.py

Verdict logic of ragregress/evaluation/regression.py on synthetic data.
No database, no CLI, no LLM: ComparisonResult objects are built by hand (diff_metrics is the
real comparator function, which is pure). Gates are injected, so the tests do not depend on the
thresholds declared by the real metrics.

Run (repo root):  python -m pytest tests/unit/test_regression.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import pytest

from ragregress.evaluation.comparator import (
    ComparisonResult,
    RowKind,
    RowView,
    RunSnapshot,
    diff_metrics,
)
from ragregress.evaluation.regression import (
    VALIDITY_MIN,
    VERDICT_EXIT_CODES,
    Gate,
    MetricStatus,
    Validity,
    Verdict,
    assess_regression,
    gates_from_registry,
    run_validity,
)

GATES = {
    "score": Gate(0.05),                               # higher is better
    "other": Gate(0.05),
    "latency": Gate(0.05, higher_is_better=False),     # lower is better
}
# "extra" is deliberately absent from GATES -> a metric with no declared threshold.


# ---------------------------------------------------------------- builders

def _ok(qa_id: str, **values: Optional[float]) -> RowView:
    return RowView(qa_id, "cat", "mode", RowKind.MEASURED, dict(values))


def _system_failure(qa_id: str, *metrics: str) -> RowView:
    # comparator semantics: no answer -> 0.0 on every applicable metric
    return RowView(qa_id, "cat", "mode", RowKind.SYSTEM_FAILURE, {m: 0.0 for m in metrics})


def _evaluator_failure(qa_id: str, *metrics: str) -> RowView:
    # comparator semantics: answer exists, judge crashed -> unmeasured
    return RowView(qa_id, "cat", "mode", RowKind.EVALUATOR_FAILURE, {m: None for m in metrics})


def _snapshot(run_id: str, rows: Iterable[RowView], applicable: Optional[Iterable[str]] = None) -> RunSnapshot:
    rows = list(rows)
    if applicable is None:
        applicable = {name for r in rows for name in r.values}
    # regression.py only reads .rows / .attempted, so a bare namespace is enough for `run`
    return RunSnapshot(
        run=SimpleNamespace(run_id=run_id),
        rows={r.qa_id: r for r in rows},
        applicable=frozenset(applicable),
    )


def _comparison(
    base_rows: Iterable[RowView],
    cand_rows: Iterable[RowView],
    ungated: Optional[Dict[str, str]] = None,
    base_applicable: Optional[Iterable[str]] = None,
    cand_applicable: Optional[Iterable[str]] = None,
) -> ComparisonResult:
    base = _snapshot("base", base_rows, base_applicable)
    cand = _snapshot("cand", cand_rows, cand_applicable)
    names = sorted(base.applicable | cand.applicable)
    common = tuple(sorted(set(base.rows) & set(cand.rows)))
    return ComparisonResult(
        baseline=base,
        candidate=cand,
        config_changes=[],
        legacy_run_ids=(),
        warnings=[],
        ungated=dict(ungated or {}),
        metrics=diff_metrics(base, cand, names),
        common_row_ids=common,
        only_in_baseline=tuple(sorted(set(base.rows) - set(cand.rows))),
        only_in_candidate=tuple(sorted(set(cand.rows) - set(base.rows))),
    )


def _pair(before: float, after: float, metric: str = "score") -> ComparisonResult:
    """One question, one metric: the delta is exactly after - before."""
    return _comparison([_ok("q1", **{metric: before})], [_ok("q1", **{metric: after})])


def _ids(n: int) -> List[str]:
    return [f"q{i:02d}" for i in range(n)]


def _metric(report, name: str):
    return next(m for m in report.metrics if m.name == name)


# ---------------------------------------------------------------- thresholds (D8-Q11, Q17)

def test_all_within_thresholds_passes():
    report = assess_regression(_pair(0.80, 0.78), GATES)
    assert report.verdict is Verdict.PASS
    assert report.gated == ("score",)
    assert _metric(report, "score").status is MetricStatus.OK


def test_improvement_passes():
    report = assess_regression(_pair(0.50, 0.90), GATES)
    assert report.verdict is Verdict.PASS


def test_worsening_of_exactly_the_threshold_is_not_regressed():
    # 0.80 - 0.75 == 0.05000000000000004 in floats: the 1e-9 tolerance must absorb it
    report = assess_regression(_pair(0.80, 0.75), GATES)
    assert report.verdict is Verdict.PASS
    assert _metric(report, "score").status is MetricStatus.OK


def test_worsening_just_over_the_threshold_fails():
    report = assess_regression(_pair(0.80, 0.7499), GATES)  # worsening 0.0501
    assert report.verdict is Verdict.FAIL
    assert _metric(report, "score").status is MetricStatus.REGRESSED
    assert any("score worsened by 0.0501" in r for r in report.reasons)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (1.00, 1.04, MetricStatus.OK),          # lower is better: +0.04 is within 0.05
        (1.00, 1.05, MetricStatus.OK),          # exactly the threshold
        (1.00, 1.06, MetricStatus.REGRESSED),   # latency went UP by more than 0.05
        (1.00, 0.20, MetricStatus.OK),          # latency dropped a lot: an improvement
    ],
)
def test_lower_is_better_direction(before, after, expected):
    report = assess_regression(_pair(before, after, metric="latency"), GATES)
    assert _metric(report, "latency").status is expected
    assert report.verdict is (Verdict.FAIL if expected is MetricStatus.REGRESSED else Verdict.PASS)


def test_higher_is_better_drop_is_not_mistaken_for_lower_is_better():
    # same numbers as the latency case: for a higher-is-better metric 1.00 -> 1.06 is an improvement
    assert assess_regression(_pair(1.00, 1.06, metric="score"), GATES).verdict is Verdict.PASS


def test_threshold_is_per_metric():
    base = [_ok("q1", score=0.9, other=0.9)]
    cand = [_ok("q1", score=0.9, other=0.8)]            # only "other" drops (by 0.1)
    report = assess_regression(_comparison(base, cand), GATES)
    assert report.verdict is Verdict.FAIL
    assert _metric(report, "score").status is MetricStatus.OK
    assert _metric(report, "other").status is MetricStatus.REGRESSED


# ---------------------------------------------------------------- validity (D8-Q3, Q8)

def _validity_case(n_measured_candidate: int, cand_score: float = 1.0, n: int = 10) -> ComparisonResult:
    ids = _ids(n)
    base = [_ok(i, score=1.0) for i in ids]
    cand = [
        _ok(i, score=cand_score) if k < n_measured_candidate else _evaluator_failure(i, "score")
        for k, i in enumerate(ids)
    ]
    return _comparison(base, cand)


def test_validity_exactly_80_percent_is_valid():
    report = assess_regression(_validity_case(8), GATES)
    assert report.validity_candidate == Validity(8, 10)
    assert report.validity_candidate.is_valid(VALIDITY_MIN)
    assert report.verdict is Verdict.PASS


def test_validity_below_80_percent_is_inconclusive():
    report = assess_regression(_validity_case(7), GATES)
    assert report.verdict is Verdict.INCONCLUSIVE
    assert any("candidate validity 7/10 = 70% < 80%" in r for r in report.reasons)


def test_baseline_validity_is_checked_too():
    ids = _ids(10)
    base = [_ok(i, score=1.0) if k < 6 else _evaluator_failure(i, "score") for k, i in enumerate(ids)]
    cand = [_ok(i, score=1.0) for i in ids]
    report = assess_regression(_comparison(base, cand), GATES)
    assert report.verdict is Verdict.INCONCLUSIVE
    assert any(r.startswith("baseline validity 6/10") for r in report.reasons)


def test_inconclusive_beats_fail_and_still_reports_the_regression():
    # 7/10 measured (invalid) AND the measured rows regressed badly
    report = assess_regression(_validity_case(7, cand_score=0.0), GATES)
    assert report.verdict is Verdict.INCONCLUSIVE
    assert _metric(report, "score").status is MetricStatus.REGRESSED
    assert any("also flagged regressed" in r for r in report.reasons)


def test_system_failure_counts_as_measured_and_scores_zero():
    ids = _ids(10)
    base = [_ok(i, score=1.0) for i in ids]
    cand = [_ok(i, score=1.0) for i in ids[:8]] + [_system_failure(i, "score") for i in ids[8:]]
    result = _comparison(base, cand)

    assert run_validity(result.candidate, ["score"]) == Validity(10, 10)

    report = assess_regression(result, GATES)
    assert report.verdict is Verdict.FAIL                  # mean 0.8 vs 1.0 -> worsened by 0.2
    assert _metric(report, "score").after == pytest.approx(0.8)
    assert _metric(report, "score").n_common == 10


def test_evaluator_failure_is_excluded_and_lowers_validity():
    ids = _ids(10)
    base = [_ok(i, score=1.0) for i in ids]
    cand = [_ok(i, score=1.0) for i in ids[:8]] + [_evaluator_failure(i, "score") for i in ids[8:]]
    result = _comparison(base, cand)

    assert run_validity(result.candidate, ["score"]) == Validity(8, 10)

    report = assess_regression(result, GATES)
    assert _metric(report, "score").n_common == 8
    assert _metric(report, "score").after == pytest.approx(1.0)   # failed rows do not drag the mean


def test_row_is_measured_only_if_every_gated_metric_has_a_value():
    ids = _ids(10)
    base = [_ok(i, score=1.0, other=1.0) for i in ids]
    cand = [_ok(i, score=1.0, other=1.0) for i in ids[:7]] + [_ok(i, score=1.0, other=None) for i in ids[7:]]
    report = assess_regression(_comparison(base, cand), GATES)
    assert report.validity_candidate == Validity(7, 10)
    assert report.verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize(
    ("measured", "attempted", "valid"),
    [(8, 10, True), (7, 10, False), (10, 10, True), (4, 5, True), (0, 5, False), (0, 0, False)],
)
def test_validity_boundary(measured, attempted, valid):
    assert Validity(measured, attempted).is_valid(VALIDITY_MIN) is valid


# ---------------------------------------------------------------- no data / empty gate set (D8-Q12, Q18)

def test_no_common_measured_rows_is_inconclusive():
    base = [_ok(i, score=0.9) for i in ("q1", "q2", "q3")]
    cand = [_ok(i, score=0.9) for i in ("q4", "q5", "q6")]       # disjoint question sets
    report = assess_regression(_comparison(base, cand), GATES)
    assert report.verdict is Verdict.INCONCLUSIVE
    assert _metric(report, "score").status is MetricStatus.NO_DATA
    assert any("score: no rows measured in both runs" in r for r in report.reasons)


def test_empty_gated_set_is_inconclusive():
    report = assess_regression(_pair(0.9, 0.9), gates={})
    assert report.verdict is Verdict.INCONCLUSIVE
    assert report.gated == ()
    assert report.reasons == ["no gated metrics in common"]


# ---------------------------------------------------------------- statuses other than ok / regressed (D8-Q15)

def test_metric_without_declared_threshold_is_not_gated_and_cannot_fail():
    base = [_ok("q1", score=0.9, extra=1.0)]
    cand = [_ok("q1", score=0.9, extra=0.0)]                     # extra collapses, but has no gate
    report = assess_regression(_comparison(base, cand), GATES)

    extra = _metric(report, "extra")
    assert extra.status is MetricStatus.NOT_GATED
    assert extra.threshold is None
    assert extra.delta == pytest.approx(-1.0)                    # still shown
    assert "extra" not in report.gated
    assert report.verdict is Verdict.PASS


def test_ungated_metric_is_shown_but_cannot_fail_or_affect_validity():
    base = [_ok("q1", score=1.0, other=0.5)]
    cand = [_ok("q1", score=None, other=0.5)]                    # score unmeasured AND would regress
    result = _comparison(base, cand, ungated={"score": "top_k differs (baseline=5, candidate=10)"})
    report = assess_regression(result, GATES)

    score = _metric(report, "score")
    assert score.status is MetricStatus.UNGATED
    assert score.note == "top_k differs (baseline=5, candidate=10)"
    assert report.gated == ("other",)
    assert report.validity_candidate == Validity(1, 1)           # only gated metrics count
    assert report.verdict is Verdict.PASS


def test_ungated_metric_with_a_real_drop_does_not_fail():
    base = [_ok("q1", score=1.0, other=0.5)]
    cand = [_ok("q1", score=0.0, other=0.5)]
    result = _comparison(base, cand, ungated={"score": "judge_model differs"})
    assert assess_regression(result, GATES).verdict is Verdict.PASS


def test_metric_absent_from_one_run_is_not_applicable():
    base = [_ok("q1", score=0.9, other=0.9)]
    cand = [_ok("q1", score=0.9)]                                # "other" never computed in candidate
    result = _comparison(base, cand)
    report = assess_regression(result, GATES)

    other = _metric(report, "other")
    assert other.status is MetricStatus.NOT_APPLICABLE
    assert other.note == "absent in candidate"
    assert report.gated == ("score",)                            # excluded from gate and validity
    assert report.verdict is Verdict.PASS


def test_metric_absent_from_the_baseline_is_not_applicable():
    base = [_ok("q1", score=0.9)]
    cand = [_ok("q1", score=0.9, other=0.9)]
    report = assess_regression(_comparison(base, cand), GATES)
    assert _metric(report, "other").note == "absent in baseline"


# ---------------------------------------------------------------- gates_from_registry (D8-Q14)

def test_gates_from_registry_skips_metrics_without_a_threshold():
    class WithThreshold:
        regression_threshold = 0.1
        higher_is_better = False

    class NoAttribute:
        pass

    class ExplicitNone:
        regression_threshold = None

    class ZeroThreshold:
        regression_threshold = 0                    # 0 is a real threshold ("no worsening allowed")

    registry = {"a": WithThreshold, "b": NoAttribute, "c": ExplicitNone, "d": ZeroThreshold}
    assert gates_from_registry(registry) == {
        "a": Gate(0.1, higher_is_better=False),
        "d": Gate(0.0, higher_is_better=True),
    }


def test_default_registry_gates_the_four_core_metrics():
    gates = gates_from_registry()
    assert {"recall_at_k", "mrr", "token_prf1", "faithfulness"} <= set(gates)
    assert all(isinstance(g, Gate) for g in gates.values())


def test_assess_regression_defaults_to_registry_gates():
    # no `gates` argument -> read from METRIC_REGISTRY; "recall_at_k" is a registered, gated metric
    report = assess_regression(_pair(0.90, 0.50, metric="recall_at_k"))
    assert report.verdict is Verdict.FAIL
    assert report.gated == ("recall_at_k",)


# ---------------------------------------------------------------- exit codes (D8-Q5, N1)

@pytest.mark.parametrize(
    ("verdict", "code"),
    [(Verdict.PASS, 0), (Verdict.FAIL, 6), (Verdict.INCONCLUSIVE, 7)],
)
def test_verdict_exit_code_mapping(verdict, code):
    assert VERDICT_EXIT_CODES[verdict] == code


def test_every_verdict_has_an_exit_code_and_none_collide_with_reserved_codes():
    assert set(VERDICT_EXIT_CODES) == set(Verdict)
    assert not {1, 2, 8} & set(VERDICT_EXIT_CODES.values())      # 1 unexpected, 2 Typer, 8 invalid input


def test_report_exit_code_follows_the_verdict():
    assert assess_regression(_pair(0.8, 0.8), GATES).exit_code == 0
    assert assess_regression(_pair(0.8, 0.5), GATES).exit_code == 6
    assert assess_regression(_pair(0.8, 0.8), gates={}).exit_code == 7