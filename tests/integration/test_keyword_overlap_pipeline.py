
import json
from pathlib import Path

from ragregress.core.config import load_run_config
from ragregress.core.runner import build_rag_system


ROOT = Path(__file__).resolve().parents[2]


def test_keyword_overlap_pipeline_retrieves_valid_passages():
    # Load the example pipeline configuration.
    config = load_run_config(
        ROOT / "configs" / "keyword_overlap.yaml"
    )

    # Build the custom pipeline.
    pipeline = build_rag_system(
        config.pipeline.identity,
        config.pipeline.params,
    )

    # Load validation questions and passages.
    split_path = ROOT / "data" / "splits" / "val.json"
    passages_path = ROOT / "data" / "passages.json"

    split = json.loads(split_path.read_text(encoding="utf-8"))
    passages = json.loads(passages_path.read_text(encoding="utf-8"))

    # Use the first validation question.
    question = split[0]["question"]

    # Collect all valid passage IDs.
    valid_ids = {p["passage_id"] for p in passages}

    # Retrieve passages.
    results = pipeline.retrieve(question, k=config.top_k)

    # Ensure retrieval returned results.
    assert results, "Pipeline returned no passages"

    # Extract retrieved IDs.
    returned_ids = {
        result.source_passage_id for result in results
    }

    # Every returned passage must exist in passages.json.
    assert returned_ids <= valid_ids
