"""
scripts/migrate_gold_passage_ids.py

One-off migration: renames the existing gold_passage_id (str) field to
gold_passage_ids (List[str]) directly on the already-saved
data/splits/{dev,val,test}.json files.

Deliberately does NOT re-run ragregress.benchmark.loader's build/sample
pipeline -- that would overwrite category (always None at construction
time) and wipe out the dev hand-tagging / val+test LLM-tagging already
done and saved to these files (see loader.py's module docstring). This
script only transforms the one field's shape, in place, preserving every
other field (qa_id, question, answer, answer_start, title, category) and
the exact qa_id set/order in each split -- no re-sampling happens.

Idempotent: a record that already has gold_passage_ids (no
gold_passage_id) is left untouched, so this is safe to run more than
once.

Usage:
    python scripts/migrate_gold_passage_ids.py
"""

import json
from pathlib import Path
from typing import List

from ragregress.benchmark.schemas import QAPair

SPLITS_DIR = Path("data") / "splits"
SPLIT_NAMES = ["dev", "val", "test"]


def migrate_split(name: str) -> None:
    path = SPLITS_DIR / f"{name}.json"
    raw: List[dict] = json.loads(path.read_text(encoding="utf-8"))

    migrated = 0
    already_done = 0

    for row in raw:
        if "gold_passage_ids" in row:
            already_done += 1
            continue
        if "gold_passage_id" not in row:
            raise KeyError(
                f"{path}: record {row.get('qa_id', '?')!r} has neither "
                f"'gold_passage_id' nor 'gold_passage_ids' -- unexpected shape, "
                f"stopping rather than guessing."
            )
        row["gold_passage_ids"] = [row.pop("gold_passage_id")]
        migrated += 1

    # Round-trip every record through the (already-updated) QAPair model to
    # confirm the migrated shape actually validates before writing anything.
    validated = [QAPair(**row) for row in raw]

    with open(path, "w", encoding="utf-8") as f:
        json.dump([q.model_dump() for q in validated], f, indent=2, ensure_ascii=False)

    print(
        f"  {name}: {migrated} migrated, {already_done} already in the new shape, "
        f"{len(validated)} total records validated and saved"
    )


def main() -> None:
    print("Migrating gold_passage_id -> gold_passage_ids in place...")
    for name in SPLIT_NAMES:
        migrate_split(name)
    print("Done. data/passages.json was not touched (Passage schema is unaffected).")


if __name__ == "__main__":
    main()