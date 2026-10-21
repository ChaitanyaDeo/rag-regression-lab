"""Day 8 / D8-Q3: classify failed rows in a stored run by failure stage (read-only)."""
import sqlite3
import sys

run_id = sys.argv[1] if len(sys.argv) > 1 else "val-11a459fe-20261017-034157"
conn = sqlite3.connect("ragregress.db")
rows = conn.execute(
    """SELECT qa_id, query_status, failure_mode,
              generation_answer IS NOT NULL AS has_answer,
              retrieved_ids_json, metric_results_json, last_error
       FROM run_query_results
       WHERE run_id = ? AND (query_status = 'failed' OR failure_mode = 'failed')""",
    (run_id,),
).fetchall()

print(f"run_id={run_id}  failed_rows={len(rows)}")
for qa_id, status, mode, has_answer, retrieved, metrics, err in rows:
    stage = "JUDGE (evaluator)" if has_answer else "RETRIEVE/GENERATE (system)"
    print(f"- {qa_id} | status={status} mode={mode} | answer={bool(has_answer)} -> {stage}")
    print(f"    retrieved_ids={retrieved[:80] if retrieved else retrieved}")
    print(f"    metric_results={metrics}")
    print(f"    last_error={(err or '')[:160]}")