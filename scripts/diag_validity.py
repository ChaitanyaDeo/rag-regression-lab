"""READ-ONLY: which rows of a run are invalid / have no faithfulness value?  python scripts/diag_validity.py [run_id]"""
import json
import math
import sqlite3
import sys
from collections import Counter

run_id = sys.argv[1] if len(sys.argv) > 1 else "val-demo-rerank-20261028-133845"
con = sqlite3.connect("file:ragregress.db?mode=ro", uri=True)
rows = con.execute(
    "SELECT qa_id, query_status, failure_mode, last_error, metric_results_json, generation_answer "
    "FROM run_query_results WHERE run_id = ?", (run_id,)).fetchall()
print("run:", run_id, "rows:", len(rows))
print("query_status counts:", dict(Counter(r[1] for r in rows)))
print("failure_mode counts:", dict(Counter(r[2] for r in rows)))
print()
for qa_id, status, mode, err, mj, ans in rows:
    res = json.loads(mj) if mj else {}
    faith = (res.get("faithfulness") or {}).get("value")
    missing = [n for n, r in res.items() if r.get("value") is None or (isinstance(r.get("value"), float) and math.isnan(r["value"]))]
    if status != "ok" or missing or not res:
        print(f"{qa_id}  status={status}  mode={mode}  missing_metrics={missing}  faithfulness={faith}")
        print(f"    last_error={str(err)[:200]!r}")
        print(f"    answer={str(ans)[:120]!r}")