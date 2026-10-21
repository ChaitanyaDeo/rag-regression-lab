"""
scripts/check_judge_failed.py -- read-only

Lists, for one stored run, how many rows carry each query_status / failure_mode, and the rows that are
`judge_failed` or `failed` (qa_id, which metrics are present, first 120 chars of last_error).

Run (repo root):
    python scripts/check_judge_failed.py --run RUN_ID [--db ragregress.db]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--db", default="ragregress.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    if not conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (args.run,)).fetchone():
        raise SystemExit(f"run not found: {args.run}")

    rows = conn.execute(
        "SELECT qa_id, query_status, failure_mode, last_error, metric_results_json "
        "FROM run_query_results WHERE run_id = ? ORDER BY qa_id",
        (args.run,),
    ).fetchall()

    print(f"run {args.run}: {len(rows)} rows")
    for (status, mode), n in sorted(Counter((r["query_status"], r["failure_mode"]) for r in rows).items(), key=str):
        print(f"  {str(status):<10} {str(mode):<18} {n}")

    flagged = [r for r in rows if r["failure_mode"] in ("judge_failed", "failed")]
    print(f"\nrows tagged judge_failed / failed: {len(flagged)}")
    for r in flagged:
        metrics = sorted(json.loads(r["metric_results_json"]).keys()) if r["metric_results_json"] else []
        err = (r["last_error"] or "").replace("\n", " ")[:120]
        print(f"  {r['qa_id']}  {r['query_status']:<7} {r['failure_mode']:<13} metrics={metrics}")
        print(f"      last_error: {err}")


if __name__ == "__main__":
    main()