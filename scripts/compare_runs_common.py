"""
scripts/compare_runs_common.py

Apples-to-apples comparison of two stored runs on the rows that were scored
(query_status = 'judged') in BOTH runs. Needed when a run has failed rows:
its aggregates silently cover fewer rows than the other run's (DEC-D7-23).

Read-only; uses only the sqlite DB (no project imports, no Ollama).

Usage:
    python scripts/compare_runs_common.py <run_id_A> <run_id_B> [--db ragregress.db]
    # A = baseline, B = candidate. Deltas are B - A.
"""

import argparse
import json
import math
import sqlite3
from collections import Counter

METRICS = ("recall_at_k", "mrr", "token_prf1", "faithfulness")


def load_judged(conn, run_id):
    rows = conn.execute(
        "select qa_id, failure_mode, metric_results_json from run_query_results "
        "where run_id = ? and query_status = 'judged'",
        (run_id,),
    ).fetchall()
    out = {}
    for qa_id, failure_mode, mj in rows:
        results = json.loads(mj) if mj else {}
        out[qa_id] = (failure_mode, {k: (v or {}).get("value") for k, v in results.items()})
    return out


def finite_mean(values):
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="baseline run_id")
    ap.add_argument("run_b", help="candidate run_id")
    ap.add_argument("--db", default="ragregress.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    for rid in (args.run_a, args.run_b):
        if not conn.execute("select 1 from runs where run_id = ?", (rid,)).fetchone():
            raise SystemExit(f"run not found: {rid}")

    a, b = load_judged(conn, args.run_a), load_judged(conn, args.run_b)
    common = sorted(set(a) & set(b))
    print(f"A = {args.run_a}: {len(a)} judged rows")
    print(f"B = {args.run_b}: {len(b)} judged rows")
    print(f"common judged rows: {len(common)}"
          f"  (only in A: {len(set(a) - set(b))}, only in B: {len(set(b) - set(a))})\n")
    if not common:
        raise SystemExit("no common judged rows")

    print(f"{'metric':<14}{'A':>9}{'B':>9}{'B-A':>9}   rows B better / worse / tie")
    for m in METRICS:
        va = [a[q][1].get(m) for q in common]
        vb = [b[q][1].get(m) for q in common]
        ma, mb = finite_mean(va), finite_mean(vb)
        better = worse = tie = 0
        for x, y in zip(va, vb):
            if not all(isinstance(v, (int, float)) and not math.isnan(v) for v in (x, y)):
                continue
            if math.isclose(x, y, abs_tol=1e-9):
                tie += 1
            elif y > x:
                better += 1
            else:
                worse += 1
        print(f"{m:<14}{ma:>9.4f}{mb:>9.4f}{mb - ma:>+9.4f}   {better} / {worse} / {tie}")

    print("\nfailure modes on common rows:")
    ca = Counter(a[q][0] for q in common)
    cb = Counter(b[q][0] for q in common)
    for mode in sorted(set(ca) | set(cb)):
        print(f"  {str(mode):<18} A={ca.get(mode, 0):<3} B={cb.get(mode, 0):<3}")


if __name__ == "__main__":
    main()