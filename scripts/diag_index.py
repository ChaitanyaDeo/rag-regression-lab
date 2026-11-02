"""
READ-ONLY diagnostic for D12 B4.4 (index 'rebuilt' when 'up to date' was expected).
Run from the repo root:  python diag_index.py
Never calls ensure_indexed / replace_all / upsert; the DB is opened read-only.
"""
import json
import sqlite3
import sys
from pathlib import Path

BASELINE_RUN = "val-demo-baseline-20261023-145024"
RERANK_RUN = sys.argv[1] if len(sys.argv) > 1 else "val-demo-rerank-20261028-133845"
CHROMA_DIR = "artifacts/chroma"
DB = "ragregress.db"


def section(title):
    print(f"\n=== {title} ===")


# 1. every collection: name, count, metadata
section("1. Chroma collections")
import chromadb

client = chromadb.PersistentClient(path=CHROMA_DIR)
for item in client.list_collections():
    name = item if isinstance(item, str) else item.name
    col = client.get_collection(name)
    print(f"{name}\n    count={col.count()}  metadata={col.metadata}")

# 2. baseline collection vs what the current code expects
section("2. Baseline collection vs current chunker/corpus")
from ragregress.core.runner import load_passages
from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig, chunk_passages
from ragregress.rag.retriever import content_fingerprint

cfg = ChunkerConfig(chunk_size=500, overlap=50)
system = BuiltinRAGSystem(cfg, embedding_model="all-MiniLM-L6-v2")
chunks = chunk_passages(load_passages(), cfg)
expected = content_fingerprint(chunks)
stored = system.retriever.stored_fingerprint()
print("expected chunks      :", len(chunks))
print("stored count         :", system.retriever.count())
print("expected fingerprint :", expected)
print("stored fingerprint   :", stored)
print("up to date           :", stored == expected and system.retriever.count() == len(chunks))

# 3. does the (re)built index retrieve like the index the baseline run used?
section("3. Vector-only top-5 now vs stored baseline retrieved_ids")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute(
    "SELECT qa_id, question, retrieved_ids_json FROM run_query_results WHERE run_id = ?",
    (BASELINE_RUN,),
).fetchall()
print("baseline rows:", len(rows))
if rows:
    print("stored retrieved_ids sample:", rows[0][2][:160])
same_set = same_order = judged = 0
diffs = []
for qa_id, question, ids_json in rows:
    if not ids_json:
        continue
    stored_ids = json.loads(ids_json)
    now = [c.chunk_id for c in system.retrieve(question, 5)]
    now_pass = [c.source_passage_id for c in system.retrieve(question, 5)]
    judged += 1
    ref = stored_ids
    match_chunk = now == ref
    match_pass = now_pass == ref
    if match_chunk or match_pass:
        same_order += 1
        same_set += 1
    elif set(now) == set(ref) or set(now_pass) == set(ref):
        same_set += 1
        diffs.append((qa_id, "same set, different order"))
    else:
        diffs.append((qa_id, "different set"))
print(f"questions compared   : {judged}")
print(f"identical top-5 order: {same_order}")
print(f"same set (any order) : {same_set}")
for d in diffs[:10]:
    print("   diff:", d)

# 4. when were the runs created?
section("4. Run timestamps")
for run_id in (BASELINE_RUN, RERANK_RUN):
    r = con.execute("SELECT created_at, config_hash FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    print(run_id, "->", tuple(r) if r else None)

section("5. Chroma files")
for p in sorted(Path(CHROMA_DIR).glob("*")):
    import datetime
    print(p.name, datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))