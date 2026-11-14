"""
tests/unit/test_generator_logger.py

The generator logs through its own module logger (post-Day 14, Q2), so its records carry
the module name and follow the package's logging setup instead of going to the root logger.

No Ollama: run_generation is replaced by a stub. No DB, no files.

Run (repo root):  python -m pytest tests/unit/test_generator_logger.py -v
"""

from __future__ import annotations

import logging

import pytest

from ragregress.rag import generator
from ragregress.rag.generator import ABSTENTION_MESSAGE, GeneratorConfig, generate_with_citations
from ragregress.rag.interface import Chunk

CONTEXT = [Chunk(chunk_id="p1::0", source_passage_id="p0000001", text="Paris is in France.")]
NAME = "ragregress.rag.generator"


def generate(monkeypatch, raw_output: str):
    monkeypatch.setattr(generator, "run_generation", lambda question, contexts, config: raw_output)
    return generate_with_citations("Where is Paris?", CONTEXT, GeneratorConfig())


def test_module_logger_has_the_module_name():
    assert generator.logger.name == NAME


def test_uncited_answer_warns_on_the_module_logger(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=NAME):
        generate(monkeypatch, "Paris is in France.")

    (record,) = [r for r in caplog.records if "no citations found" in r.getMessage()]
    assert record.name == NAME
    assert record.levelno == logging.WARNING
    assert "Where is Paris?" in record.getMessage()


@pytest.mark.parametrize("raw_output", ["Paris is in France [source: p0000001].", ABSTENTION_MESSAGE])
def test_cited_or_abstaining_answer_does_not_warn(monkeypatch, caplog, raw_output):
    with caplog.at_level(logging.WARNING, logger=NAME):
        generate(monkeypatch, raw_output)
    assert not [r for r in caplog.records if "no citations found" in r.getMessage()]