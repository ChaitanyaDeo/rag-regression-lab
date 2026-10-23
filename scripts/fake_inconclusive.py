"""
scripts/fake_inconclusive.py

D9-B8: make a scratch copy of ragregress.db in which the candidate run is INCONCLUSIVE,
so the report's warning banner can be checked on real data. The real DB is never modified.

It marks N judged candidate rows as 'failed' but keeps their answers, so the comparator treats
them as evaluator failures (unmeasured). Candidate validity: 48/50 -> 38/50 = 76% < 80%.

Run (repo root):
    python scripts/fake_inconclusive.py
    ragregress report --db reports/fake_inconclusive.db --baseline val-5352aebf-20261017-031708 --candidate val-11a459fe-b9rerun --out reports/fake_inconclusive.html --open
"""

import shutil
import sqlite3
from pathlib import Path

SOURCE = Path("ragregress.db")
SCRATCH = Path("reports/fake_inconclusive.db")
CANDIDATE = "val-11a459fe-b9rerun"
N_ROWS = 10

SCRATCH.parent.mkdir(exist_ok=True)
shutil.copyfile(SOURCE, SCRATCH)

with sqlite3.connect(SCRATCH) as conn:
    ids = [r[0] for r in conn.execute(
        "SELECT qa_id FROM run_query_results WHERE run_id = ? AND query_status = 'judged' "
        "AND failure_mode != 'judge_failed' ORDER BY qa_id LIMIT ?", (CANDIDATE, N_ROWS))]
    conn.executemany(
        "UPDATE run_query_results SET query_status = 'failed' WHERE run_id = ? AND qa_id = ?",
        [(CANDIDATE, qa_id) for qa_id in ids])

print(f"{SCRATCH}: marked {len(ids)} candidate rows as failed (answers kept); {SOURCE} untouched")