"""
Ad-hoc integration check (not part of the project's own smoke tests):
exercises the real Pipeline (with storage wiring) end-to-end against a
fake RAGSystem and a fake StructuredLLM, then inspects the resulting
SQLite rows via Repository.

Uses fakes for BOTH rag_system and the Faithfulness judge llm so this
runs instantly and offline -- no real retrieval index, no real model
call. It is checking the Pipeline -> Repository -> SQLite wiring, not
metric quality or retrieval/generation correctness (those already have
their own tests/scripts elsewhere).

Writes to a real file at the project root -- integration_check.db,
named distinctly from ragregress.db (the real pipeline's database) so
the two are never confused -- rather than a tempfile, so the resulting
rows can be inspected afterward (e.g. `sqlite3 integration_check.db`)
instead of vanishing the moment the script exits. Deleted and
recreated fresh on every run, so re-running never trips the "run_id
already exists" path or leaves stale rows from a previous run.

Not meant to replace scripts/smoke_test_pipeline.py or
scripts/smoke_test_storage.py.
"""

from pathlib import Path

from ragregress.benchmark.schemas import QAPair
from ragregress.evaluation.metrics import RecallAtK, TokenPRF1, Faithfulness
from ragregress.evaluation.metrics.faithfulness.utils import StatementGeneratorOutput, NLIStatementOutput, StatementFaithfulnessAnswer
from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.pipeline.pipeline import Pipeline
from ragregress.storage.repositories import Repository

K = 3


class FakeRAGSystem:
    def retrieve(self, question, k):
        return [Chunk(chunk_id="c1", source_passage_id="p1", text="Newton on gravity...", metadata={})]

    def generate(self, question, chunks):
        return GenerationResult(raw_output="raw", answer="Newton's law of gravitation", citations=[])


class FakeStructuredLLM:
    """Stands in for a real StructuredLLM so Faithfulness doesn't need a
    live model. Returns one canned statement, marked faithful, regardless
    of the prompt -- schema-shape is what this check cares about, not
    judgment quality."""

    def generate_structured(self, prompt, schema):
        if schema is StatementGeneratorOutput:
            return StatementGeneratorOutput(statements=["Newton formulated the law of gravitation."])
        if schema is NLIStatementOutput:
            return NLIStatementOutput(statements=[
                StatementFaithfulnessAnswer(
                    statement="Newton formulated the law of gravitation.",
                    reason="supported by context",
                    verdict=1,
                )
            ])
        raise ValueError(f"FakeStructuredLLM: unexpected schema {schema!r}")


# scripts/ -> parent is the project root, regardless of the cwd this is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "integration_check.db"


def main():
    # Fresh file every run -- avoids re-inserting "integration-run-1" into a
    # run_id that already exists from a previous invocation.
    if DB_PATH.exists():
        DB_PATH.unlink()

    pipeline = Pipeline(
        rag_system=FakeRAGSystem(),
        metrics=[RecallAtK(k=K), TokenPRF1(), Faithfulness(llm=FakeStructuredLLM())],
        k=K,
        max_workers=1,
        db_path=DB_PATH,
    )

    qa_pairs = [
        QAPair(
            qa_id="qa-1",
            question="Who formulated the law of gravitation?",
            answer=["Newton's law of gravitation"],
            answer_start=[0],
            gold_passage_ids=["p1"],
            title="Force",
        ),
        QAPair(
            qa_id="qa-2",
            question="second question",
            answer=["something else"],
            answer_start=[0],
            gold_passage_ids=["p1"],
            title="Force",
        ),
    ]

    result = pipeline.run(qa_pairs=qa_pairs, run_id="integration-run-1")
    print("\n--- graph.invoke() result ---")
    print("run_status:", result.get("run_status"))
    print("aggregated_metrics:", result.get("aggregated_metrics"))
    print("per_query_results count:", len(result.get("per_query_results", [])))

    repo = Repository(DB_PATH)
    run = repo.get_run("integration-run-1")
    metrics = repo.get_metrics("integration-run-1")
    query_results = repo.get_query_results("integration-run-1")

    print("\n--- SQLite contents ---")
    print("Run row:", run)
    print("Metrics rows:", metrics)
    print("QueryResult rows:")
    for qr in query_results:
        print(" ", qr)

    assert run is not None
    assert run.run_status == "healthy"
    assert len(metrics) == 3
    assert len(query_results) == 2
    assert {qr.qa_id for qr in query_results} == {"qa-1", "qa-2"}
    assert all(qr.retrieved_ids == [] for qr in query_results), "chunks intentionally not persisted yet"
    assert all(qr.generation_answer == "Newton's law of gravitation" for qr in query_results)
    assert all(qr.metric_results for qr in query_results)

    print(f"\n=== Integration check PASSED: Pipeline -> Repository -> SQLite wiring works end-to-end ===")
    print(f"(rows written to {DB_PATH} -- inspect with: sqlite3 {DB_PATH.name})")


if __name__ == "__main__":
    main()