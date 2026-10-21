"""fact-finding (read-only): what do `category` and `failure_mode` look like in stored runs?

Usage (repo root, venv active):
    python scripts/check_groups.py <run_id> [<run_id> ...] [--db ragregress.db]
"""
import argparse
from collections import Counter

from ragregress.storage.repositories import Repository


def show(title, counter):
    print(f"  {title}:")
    for key, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"    {str(key):<28}{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--db", default="ragregress.db")
    args = ap.parse_args()

    repo = Repository(args.db)
    for run_id in args.run_ids:
        rows = repo.get_query_results(run_id)
        print(f"\n{run_id}  ({len(rows)} rows)")
        show("category", Counter(r.category for r in rows))
        show("failure_mode", Counter(r.failure_mode for r in rows))
        show("query_status", Counter(r.query_status for r in rows))
        show(
            "(status, failure_mode)",
            Counter((r.query_status, r.failure_mode) for r in rows),
        )


if __name__ == "__main__":
    main()