
"""
ragregress/benchmark/loader.py

Loader for benchmark datasets

IMP : This works as a mapping from dataset to schema: it changes based dataset'

Loads SQuAD (rajpurkar/squad), builds the deduplicated passage corpus from
the validation split, links every question to its exact
gold passage, and samples the three fixed dev/val/test splits with
EVAL_SEED=42 

Run directly to build and save everything to disk:
 
    python -m ragregress.benchmark.loader

    Produces:
    data/passages.json      — full deduplicated corpus (all unique contexts
                               in the validation split)
    data/splits/dev.json    — 15 questions
    data/splits/val.json    — 50 questions
    data/splits/test.json   — 200 questions

"""

from __future__ import annotations
 
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
 
from datasets import load_dataset

from ragregress.benchmark.schemas import Passage, QAPair

EVAL_SEED = 42

DEV_SIZE = 15
VAL_SIZE = 50
TEST_SIZE = 200     # Capped, not acutal full validation split size (10570 questions)

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"

def load_raw_validation():
    """Loads the SQuAD validation split only for SQuAD"""
    squad = load_dataset("rajpurkar/squad")
    return squad["validation"]

def build_corpus_and_qa_pairs(validation_split) -> Tuple[List[Passage], List[QAPair]]:
    """Deduplicates `context` values into a passage corpus (sequential
    passage_id assigned in first-seen order), and builds QAPair objects
    linked to their exact gold passage.
    """
    context_to_passage_id: Dict[str, str] = {}
    passages: List[Passage] = []
    qa_pairs: List[QAPair] = []

    for row in validation_split:
        context = row["context"]
        title = row["title"]

        if context not in context_to_passage_id:
            passage_id = f"p{len(passages):07d}"

            context_to_passage_id[context] = passage_id
            passages.append(Passage(passage_id = passage_id, title = title, text = context))

        gold_passage_id = context_to_passage_id[context]

        qa_pairs.append(
            QAPair(
                qa_id = row["id"],
                question = row["question"],
                answer=row["answers"]["text"],
                answer_start = row["answers"]["answer_start"],
                gold_passage_ids = [gold_passage_id],
                title = title,
                category = None
            )
        )

    return passages, qa_pairs

def sample_splits(qa_pairs: List[QAPair]) -> Dict[str, List[QAPair]]:
    """Samples dev/val/test once with EVAL_SEED=42. All three are subsets of
    the same frozen question set — not resampled, and may overlap (this
    matches the project's original split semantics: three fixed-size
    views at different iteration speeds, not an ML train/val/test split)
    """

    rng = random.Random(EVAL_SEED)

    dev = rng.sample(qa_pairs, DEV_SIZE)
    val = rng.sample(qa_pairs, VAL_SIZE)
    test = rng.sample(qa_pairs, TEST_SIZE)

    return {"dev": dev, "val": val, "test": test}


def save_json(obj_list, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([o.model_dump() for o in obj_list], f, indent=2, ensure_ascii=False)


def main() -> None:
    print("Loading SQuAD validation split...")
    raw_validation = load_raw_validation()
    print(f"  {raw_validation.num_rows} raw QA rows")

    print("Building deduplicated corpus + QA pairs...")
    passages, qa_pairs = build_corpus_and_qa_pairs(raw_validation)
    print(f"  {len(passages)} unique passages")
    print(f"  {len(qa_pairs)} QA pairs, every one linked to an exact gold passage")

    print(f"Sampling dev/val/test with EVAL_SEED={EVAL_SEED}...")
    splits = sample_splits(qa_pairs)
    for name, split in splits.items():
        print(f"  {name}: {len(split)} questions")

    print("Saving to disk...")
    save_json(passages, DATA_DIR / "passages.json")
    for name, split in splits.items():
        save_json(split, SPLITS_DIR / f"{name}.json")
 
    print("Done.")


if __name__ == "__main__":
    main()

    

