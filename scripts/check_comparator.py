"""check: exercise ragregress.evaluation.comparator on stored runs (read-only).

Usage (repo root, venv active):
    python scripts/check_comparator.py <baseline_run_id> <candidate_run_id> [--db ragregress.db]

Prints what compare_runs() computed. No verdict yet (that is B2).
Exit-8 guard errors are printed, not raised, so you can try a dev run here too.
"""
import argparse

from ragregress.core.errors import ComparisonInputError
from ragregress.evaluation.comparator import compare_runs
from ragregress.storage.repositories import Repository


def fmt(x):
    return "   n/a" if x is None else f"{x:7.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--db", default="ragregress.db")
    args = ap.parse_args()

    try:
        res = compare_runs(Repository(args.db), args.baseline, args.candidate)
    except ComparisonInputError as exc:
        print(f"REJECTED (exit_code={exc.exit_code}): {type(exc).__name__}: {exc}")
        return

    print(f"baseline : {res.baseline.run.run_id}  rows={res.baseline.attempted}  applicable={sorted(res.baseline.applicable)}")
    print(f"candidate: {res.candidate.run.run_id}  rows={res.candidate.attempted}  applicable={sorted(res.candidate.applicable)}")
    print(f"common rows={len(res.common_row_ids)}  only_base={len(res.only_in_baseline)}  only_cand={len(res.only_in_candidate)}")

    kinds = {}
    for label, snap in (("baseline", res.baseline), ("candidate", res.candidate)):
        counts = {}
        for row in snap.rows.values():
            counts[row.kind.value] = counts.get(row.kind.value, 0) + 1
        kinds[label] = counts
        print(f"row kinds {label}: {counts}")

    print("\nconfig changes:")
    for c in res.config_changes:
        print(f"  {c.key}: {c.before} -> {c.after}")
    if not res.config_changes:
        print("  (none)")

    print("\nwarnings:", res.warnings or "(none)")
    print("ungated :", res.ungated or "(none)")

    print(f"\n{'metric':<14}{'before':>8}{'after':>9}{'delta':>9}  n   better/worse/tie  applicable(base,cand)")
    for m in res.metrics:
        d = "     n/a" if m.delta is None else f"{m.delta:+8.4f}"
        print(f"{m.name:<14}{fmt(m.before)}{fmt(m.after)} {d}  {m.n_common:<3} {m.better}/{m.worse}/{m.tie}"
              f"             ({m.applicable_base},{m.applicable_cand})")


if __name__ == "__main__":
    main()