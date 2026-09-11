"""
Fixed-size character chunking.

Internal to BuiltinRAGSystem only — this is the lab's own reference
pipeline's chunking step, not part of the public RAGSystem contract. A
user plugging in their own pipeline never imports this module; they
implement RAGSystem (see interface.py) and chunk however they want inside
their own retrieve().

Deliberately hand-rolled rather than LangChain's RecursiveCharacterTextSplitter:
chunk size is the variable under test across configs, and this project wants
deterministic, easily-reasoned-about chunk boundaries rather than
recursive-separator behavior.


"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import CharacterTextSplitter
 
from ragregress.benchmark.schemas import Passage
from ragregress.rag.interface import Chunk


@dataclass
class ChunkerConfig:
    chunk_size: int = 500
    overlap: int = 50

    # Validation (overlap < chunk_size, etc.) is handled by
    # CharacterTextSplitter's own constructor, which raises if
    # chunk_overlap >= chunk_size.


def chunk_passage(passage: Passage, config: ChunkerConfig)-> List[Chunk]:
    """Split one passage text into overlapping fixed-size chunks """

    text = passage.text

    if not text:
        return []

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=config.chunk_size,
        chunk_overlap=config.overlap,
        length_function=len,
    )

    pieces = splitter.split_text(text)

    chunks: List[Chunk] = []

    for idx, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{passage.passage_id}::{idx}",
                source_passage_id=passage.passage_id,
                text=piece,
                metadata={"title": passage.title},
            )
        )

    return chunks

def chunk_passages(passage: List[Passage], config: ChunkerConfig)-> List[Chunk]:
    out: List[Chunk] = []

    for p in passage:
        out.extend(chunk_passage(p, config))

    return out
