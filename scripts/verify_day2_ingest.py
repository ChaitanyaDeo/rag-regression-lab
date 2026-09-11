"""
verification script

Ingests the full passage corpus into Chroma via BuiltinRAGSystem (baseline
config: chunk_size=500, MiniLM), retrieves top-5 for a sample query, and
asserts every returned Chunk.source_passage_id resolves to a real id in
data/passages.json (the contract check from the RAGSystem interface).

Run twice to also confirm the Chroma collection persists across a process
restart (second run should ingest 0 *new* points if add_texts is
idempotent on id, or you can wipe artifacts/chroma/ between runs to force
a clean rebuild — see note at bottom).

Usage:
    python scripts/verify_day2_ingest.py

"""

import json
from pathlib import Path
 
from ragregress.benchmark.schemas import Passage
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig

DATA_DIR = Path("data")
SAMPLE_QUERY = "What is the capital of France?"

def load_passages() -> list[Passage]:
    raw = json.loads((DATA_DIR / "passages.json").read_text(encoding="utf-8"))
    return [Passage(**p) for p in raw]

def main() -> None:
    passages = load_passages()
    valid_ids = {p.passage_id for p in passages}
    print(f"Loaded {len(passages)} passages from data/passages.json")
 
    system = BuiltinRAGSystem(chunker_config=ChunkerConfig(chunk_size=500, overlap=50))
    n_chunks = system.ingest(passages)
    print(f"Ingested {n_chunks} chunks into collection "
          f"'{system.retriever.collection_name}'")
 
    results = system.retrieve(SAMPLE_QUERY, k=5)
    print(f"\nRetrieved {len(results)} chunks for: {SAMPLE_QUERY!r}\n")
 
    for r in results:
        ok = r.source_passage_id in valid_ids
        status = "OK" if ok else "INVALID"
        print(f"  [{status}] chunk_id={r.chunk_id!r} source_passage_id={r.source_passage_id!r}")
        print(f"         text preview: {r.text[:120]!r}")
        if not ok:
            raise AssertionError(
                f"source_passage_id {r.source_passage_id!r} does not resolve to any "
                f"passage in data/passages.json"
            )
 
    print("\nAll source_passage_id values verified against data/passages.json.")
    print("\nNote: re-run this script (new process) to confirm the Chroma collection")
    print("persists across a restart — the second run's retrieve() should still work")
    print("without re-ingesting from scratch being required for retrieval to succeed.")

if __name__ == "__main__":
    main()