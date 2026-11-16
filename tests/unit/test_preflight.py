"""
tests/unit/test_preflight.py

Tests for RAG pipeline preflight validation.

"""

import json
import logging
from pathlib import Path

import pytest

from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.rag.preflight import run_preflight


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

@pytest.fixture
def passages():
    return json.loads((FIXTURES / "passages_5doc.json").read_text(encoding="utf-8"))

@pytest.fixture
def qa_pairs():
    return json.loads((FIXTURES / "qa_5doc.json").read_text(encoding="utf-8"))

@pytest.fixture
def corpus_ids(passages):
    return {p["passage_id"] for p in passages}


class FakeRAG:
    """Configurable fake RAG system for preflight tests."""

    def __init__(self, chunks=None, answer="Test answer", retrieve_error=None, generate_error=None, invalid_generation=False,):
        self.chunks = chunks if chunks is not None else []
        self.answer = answer
        self.retrieve_error = retrieve_error
        self.generate_error = generate_error
        self.invalid_generation = invalid_generation

    def retrieve(self, query, k):
        if self.retrieve_error:
            raise self.retrieve_error
        return self.chunks

    def generate(self, query, contexts):
        if self.generate_error:
            raise self.generate_error
        if self.invalid_generation:
            return "Invalid generation result"
        return GenerationResult(answer=self.answer)

def make_chunk(passage_id="p0000004", text="Example passage"):
    return Chunk(
        chunk_id=f"{passage_id}::0",
        source_passage_id=passage_id,
        text=text,
    )

def run(system, qa_pairs, corpus_ids=None, k=5):
    return run_preflight(
        system=system,
        qa_pairs=qa_pairs,
        k=k,
        corpus_ids=corpus_ids,
    )

def findings_by_check(result):
    return {finding.check: finding for finding in result.findings}


def test_conforming_pipeline_has_no_findings(qa_pairs, corpus_ids):
    system = FakeRAG(chunks=[make_chunk(pid) for pid in corpus_ids][:2])

    result = run(system, qa_pairs[:1], corpus_ids)

    assert result.findings == []
    assert result.questions[0].status == "checked"


def test_retrieve_returns_non_list(qa_pairs, corpus_ids):
    class InvalidRAG(FakeRAG):
        def retrieve(self, query, k):
            return "not a list"

    result = run(InvalidRAG(), qa_pairs[:1], corpus_ids)

    finding = findings_by_check(result)["retrieve_return_type"]
    assert finding.severity == "error"


def test_more_than_k_chunks(qa_pairs, corpus_ids):
    chunks = [make_chunk(pid) for pid in list(corpus_ids)[:3]]

    result = run(FakeRAG(chunks=chunks), qa_pairs[:1], corpus_ids, k=2)

    finding = findings_by_check(result)["too_many_chunks"]
    assert finding.severity == "error"


def test_zero_chunks(qa_pairs, corpus_ids):
    result = run(FakeRAG(chunks=[]), qa_pairs[:1], corpus_ids)

    finding = findings_by_check(result)["zero_chunks"]
    assert finding.severity == "error"


def test_unknown_source_passage_id(qa_pairs, corpus_ids):
    result = run(FakeRAG(chunks=[make_chunk("unknown_id")]), qa_pairs[:1], corpus_ids,)

    finding = findings_by_check(result)["unknown_passage_id"]
    assert finding.severity == "error"


def test_corpus_id_check_skipped_when_ids_omitted(qa_pairs):
    result = run(FakeRAG(chunks=[make_chunk("unknown_id")]), qa_pairs[:1],corpus_ids=None)

    assert "unknown_passage_id" not in findings_by_check(result)


def test_invalid_generation_result(qa_pairs, corpus_ids):
    result = run(FakeRAG(chunks=[make_chunk()], invalid_generation=True,), qa_pairs[:1],corpus_ids,)

    finding = findings_by_check(result)["generate_return_type"]
    assert finding.severity == "error"

@pytest.mark.parametrize("answer", ["", "   ", "\n\t"])
def test_empty_answer(qa_pairs, corpus_ids, answer):
    result = run(
        FakeRAG(chunks=[make_chunk()], answer=answer),
        qa_pairs[:1],
        corpus_ids,
    )

    finding = findings_by_check(result)["empty_answer"]
    assert finding.severity == "error"


def test_empty_chunk_text_is_warning(qa_pairs, corpus_ids):
    result = run(
        FakeRAG(chunks=[make_chunk(text="  ")]),
        qa_pairs[:1],
        corpus_ids,
    )

    finding = findings_by_check(result)["empty_chunk_text"]
    assert finding.severity == "warning"


def test_duplicate_source_passage_id_is_warning(qa_pairs, corpus_ids):
    result = run(
        FakeRAG(
            chunks=[
                make_chunk("p0000004"),
                make_chunk("p0000004"),
            ]
        ),
        qa_pairs[:1],
        corpus_ids,
    )

    finding = findings_by_check(result)["duplicate_source_passage_id"]
    assert finding.severity == "warning"


def test_retrieve_exception_is_recorded_not_a_finding(qa_pairs, corpus_ids):
    system = FakeRAG(
        retrieve_error=RuntimeError("retrieval failed")
    )

    result = run(system, qa_pairs[:1], corpus_ids)

    assert result.findings == []
    assert result.questions[0].status == "raised"
    assert "retrieval failed" in result.questions[0].exception


def test_generate_exception_is_recorded_not_a_finding(qa_pairs, corpus_ids):
    system = FakeRAG(
        chunks=[make_chunk()],
        generate_error=RuntimeError("generation failed"),
    )

    result = run(system, qa_pairs[:1], corpus_ids)

    assert result.findings == []
    assert result.questions[0].status == "raised"
    assert "generation failed" in result.questions[0].exception


def test_generate_skipped_when_retrieval_fails(qa_pairs, corpus_ids):
    class TrackingRAG(FakeRAG):
        generate_called = False

        def generate(self, query, contexts):
            self.generate_called = True
            return GenerationResult(answer="Test answer")

    system = TrackingRAG(
        retrieve_error=RuntimeError("retrieval failed")
    )

    run(system, qa_pairs[:1], corpus_ids)

    assert system.generate_called is False


def test_preflight_does_not_log(qa_pairs, corpus_ids, caplog):
    system = FakeRAG(chunks=[make_chunk()])

    with caplog.at_level(logging.DEBUG):
        run(system, qa_pairs[:1], corpus_ids)

    assert caplog.records == []