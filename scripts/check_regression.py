"""check: comparator + regression verdict on two stored runs (read-only, no CLI yet).

Usage (repo root, venv active):
    python scripts/check_regression.py <baseline_run_id> <candidate_run_id> [--db ragregress.db]
"""
import argparse

from ragregress.core.errors import ComparisonInputError
from ragregress.evaluation.comparator import compare_runs
from ragregress.evaluation.regression import assess_regression
from ragregress.storage.repositories import Repository


def fmt(x):
    return "     n/a" if x is None else f"{x:8.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--db", default="ragregress.db")
    args = ap.parse_args()

    try:
        result = compare_runs(Repository(args.db), args.baseline, args.candidate)
    except ComparisonInputError as exc:
        print(f"REJECTED (exit_code={exc.exit_code}): {type(exc).__name__}: {exc}")
        return

    report = assess_regression(result)

    vb, vc = report.validity_baseline, report.validity_candidate
    print(f"baseline : {args.baseline}")
    print(f"candidate: {args.candidate}")
    print(f"validity : baseline {vb.measured}/{vb.attempted} ({vb.ratio:.0%}) | candidate {vc.measured}/{vc.attempted} ({vc.ratio:.0%})")
    print(f"gated    : {list(report.gated)}")
    for w in result.warnings:
        print(f"warning  : {w}")

    print(f"\n{'metric':<14}{'before':>9}{'after':>9}{'delta':>10}  {'n':<4}{'threshold':<10}status")
    for m in report.metrics:
        thr = "-" if m.threshold is None else f"{m.threshold:.2f}"
        d = "      n/a" if m.delta is None else f"{m.delta:+9.4f}"
        note = f"  ({m.note})" if m.note else ""
        print(f"{m.name:<14}{fmt(m.before)}{fmt(m.after)} {d}  {m.n_common:<4}{thr:<10}{m.status.value}{note}")

    print(f"\nVERDICT: {report.verdict.value}  (exit code would be {report.exit_code})")
    for r in report.reasons:
        print(f"  - {r}")


if __name__ == "__main__":
    main()