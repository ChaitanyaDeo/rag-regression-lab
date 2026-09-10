'''
scripts/suggest_categories.py

LLM-assisted CRAG category tagging for the val/test splits.
dev is hand-tagged separately.

Uses qwen2.5:3b-instruct: via a plain Ollama HTTP call
this is a cheap classification task, not a faithfulness judgment, so the
dev-profile-speed model is the right fit, not the 7B judge.

Usage:
    python scripts/suggest_categories.py --split dev
    python scripts/suggest_categories.py --split val
    python scripts/suggest_categories.py --split test

Writes the category back into the same split file in place.

NOTE: For user defined categories
'''

from __future__ import annotations
 
import argparse
import json
import re
from pathlib import Path

import requests
 
from ragregress.benchmark.schemas import CRAG_CATEGORY_DEFINITIONS

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b-instruct"
 
SPLITS_DIR = Path("data/splits")
PASSAGES_PATH = Path("data/passages.json")

PROMPT_TEMPLATE = """You are a strict question classifier.
 
Choose exactly ONE category from these allowed categories:
 
{category_block}
 
Passage:
{passage}
 
Question:
{question}
 
- Return ONLY the category name.
- The category name is the text BEFORE the ":" on each category line.
- Use the category definition, and the passage content, only to determine which category best matches the question.
- The output must EXACTLY match one of the category names provided.
- Do not output the definition.
- Do not output the ":".
- Do not add explanations, punctuation, quotes, or any other text.
- Do not output multiple categories.
- Preserve the category name exactly as provided, including capitalization, spaces, hyphens, and underscores.
"""

def load_category_definitions(categories_file: str | None) -> str:
    """
    Returns {category_name: definition}. Defaults to CRAG's 8 types unless
    a user-supplied --categories-file JSON is given, in which case that
    fully replaces the default taxonomy (not merged with it).
    """

    if categories_file is None:
        return CRAG_CATEGORY_DEFINITIONS

    with open(categories_file, "r", encoding="utf-8") as f:
        user_categories = json.load(f)

    if not isinstance(user_categories, dict) or not user_categories:
        raise ValueError(
            f"{categories_file} must contain a non-empty JSON object of "
            '{"category_name": "definition"} pairs'
        )
    return user_categories

def load_passage_lookup() -> Dict[str, str]:
    """passage_id -> text, for including gold context in the prompt."""
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)
    return {p["passage_id"]: p["text"] for p in passages}




def build_category_block(definitions: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {definition}" for name, definition in definitions.items())

def classify_question(question: str, passage: str, category_definitions: dict[str,str], category_block: str) -> str:
    prompt = PROMPT_TEMPLATE.format(category_block=category_block, passage=passage, question=question)

    response = requests.post(
        OLLAMA_URL,
        json = {"model": MODEL, "prompt" : prompt, "stream": False, "options": {"temperature": 0}},
        timeout = 30,
    )

    response.raise_for_status()

    raw = response.json()["response"].strip()

    if raw in category_definitions:
        return raw

    # Fell through without a clean match -- flag it rather than guessing.
    return "UNRESOLVED:" + re.sub(r"\s+", " ", raw)[:50]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["dev", "val", "test"])
    parser.add_argument(
        "--categories-file",
        default=None,
        help="Optional path to a JSON file of {category_name: definition} pairs. "
        "Defaults to CRAG's 8 types if omitted.",
    )
    args = parser.parse_args()

    definitions = load_category_definitions(args.categories_file)

    category_block = build_category_block(definitions)

    passage_lookup = load_passage_lookup()

    print(f"Using {len(definitions)} categories: {', '.join(definitions)}")

    path = SPLITS_DIR/f"{args.split}.json"

    with open(path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    unresolved = 0

    for i, qa in enumerate(qa_pairs):
        passage_text = passage_lookup[qa["gold_passage_id"]]
        question = qa["question"]
        category = classify_question(question, passage_text, definitions, category_block)
        qa["category"] = category

        if category.startswith("UNRESOLVED:"):
            unresolved += 1

        print(f"[{i + 1}/{len(qa_pairs)}] {qa['question'][:60]!r} -> {category}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(qa_pairs)} tagged, {unresolved} unresolved (flagged for manual review).")


if __name__ == "__main__":
    main()



