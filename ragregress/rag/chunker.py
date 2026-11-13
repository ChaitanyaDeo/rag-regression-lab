"""
Chunking strategies: fixed (the default), recursive, sentence, sliding_window.

Internal to BuiltinRAGSystem only — this is the lab's own reference
pipeline's chunking step, not part of the public RAGSystem contract. A
user plugging in their own pipeline never imports this module; they
implement RAGSystem (see interface.py) and chunk however they want inside
their own retrieve().

The chunking strategy is a variable under test, like chunk size: pipeline.params.chunking selects it
(absent = fixed, the original fixed-size character windows). Every strategy is deterministic and
returns exact substrings of the passage; STRATEGIES lists them.


"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
 
from ragregress.benchmark.schemas import Passage
from ragregress.rag.interface import Chunk


DEFAULT_STRATEGY = "fixed"

@dataclass
class ChunkerConfig:
    chunk_size: int = 500
    overlap: int = 50
    strategy: str = DEFAULT_STRATEGY

    options: Dict[str, int] = field(default_factory=dict)    # the strategy's own settings (see STRATEGIES)

    
    def __post_init__(self):
        # CharacterTextSplitter only rejects overlap > chunk_size; overlap == chunk_size
        # is accepted and silently advances one character per chunk.
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        _validate_strategy(self)


# ---------------------------------------------------------------------- strategies

# A strategy turns one passage text into a list of chunk texts. 
# All of them return exact substrings of the passage, no longer than chunk_size 
# (one word longer than chunk_size is cut, as there is no better place to cut it).

Span = Tuple[int, int]

# A sentence ends at . ! or ? followed by whitespace and then a capital letter, a digit, a quote or a
# bracket; a blank line also ends one. Abbreviations ("St. Gallen", "Dr. Smith") are therefore split
# too: simple and deterministic, not linguistically exact.
_SENTENCE_BREAK = re.compile(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9\"'(\[“‘])")


def _split_fixed(text:str, config:"ChunkerConfig") -> List[Chunk]:
    """Fixed Chunking - Fixed window of chunk_size characters """

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size = config.chunk_size,
        chunk_overlap=config.overlap,
        length_function=len,
    )

    return splitter.split_text(text)


def _split_recursive(text:str, config: "ChunkerConfig") -> List[str]:
    """Recursive Chunking - Break at the largest boundary that fits: blank line, then line, then space (never mid-word)."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

# ---------------------------------------- Sentence Chunking
def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_spans(text:str)->List[Span]:
    spans: List[Span] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        spans.append(_trim(text, start, match.start()))
        start = match.end()
    spans.append(_trim(text, start, len(text)))
    return [(a, b) for a, b in spans if a < b]


def _fit_spans(text: str, spans: List[Span], size: int) -> List[Span]:
    """Cut a sentence longer than `size` at its last space that fits (a hard cut if it has none)."""
    out: List[Span] = []
    for start, end in spans:
        while end - start > size:
            cut = start + size
            while cut > start and not text[cut].isspace():      # back to the last whitespace that fits
                cut -= 1
            if cut == start:                                     # no whitespace at all: cut the word
                cut = start + size
            out.append(_trim(text, start, cut))
            start = cut
            while start < end and text[start].isspace():
                start += 1
        if start < end:
            out.append((start, end))
    return out

def _split_sentence(text: str, config: "ChunkerConfig") -> List[str]:
    """Pack whole sentences up to chunk_size. The next chunk starts with the last sentences of the
    previous one that fit in `overlap` characters (none, if even one is longer than that).
    """
    spans = _fit_spans(text, _sentence_spans(text), config.chunk_size)
    chunks: List[str] = []
    current: List[Span] = []

    for span in spans:
        if current and span[1] - current[0][0] > config.chunk_size:
            chunks.append(text[current[0][0]:current[-1][1]])
            kept: List[Span] = []
            for earlier in reversed(current):       # newest first: carry sentences while they fit in `overlap` and with the new one
                if current[-1][1] - earlier[0] > config.overlap or span[1] - earlier[0] > config.chunk_size:
                    break
                kept.append(earlier)
            current = kept[::-1]
        current.append(span)
    if current:
        chunks.append(text[current[0][0]:current[-1][1]])
    return chunks

# ------------------------ Sliding window chunking
def _split_sliding_window(text: str, config: "ChunkerConfig") -> List[str]:
    """`window` consecutive sentences, moving on by `stride` sentences. A window longer than chunk_size
    loses sentences from its end (at least one stays); the next window then starts only as far on as
    the sentences actually taken, so no sentence is skipped."""
    units = _fit_spans(text, _sentence_spans(text), config.chunk_size)
    window, stride = config.options["window"], config.options["stride"]
    chunks: List[str] = []
    i = 0
    while i < len(units):
        take = min(window, len(units) - i)
        while take > 1 and units[i + take - 1][1] - units[i][0] > config.chunk_size:
            take -= 1
        chunks.append(text[units[i][0]:units[i + take - 1][1]])
        if i + take >= len(units):
            break
        i += min(stride, take)
    return chunks


def _check_sliding_window(config: "ChunkerConfig") -> None:
    if config.options["stride"] > config.options["window"]:
        raise ValueError(
            f"stride ({config.options['stride']}) must not be larger than window ({config.options['window']})"
        )

# -----------------------
@dataclass(frozen=True)
class Strategy:
    split: Callable[[str, "ChunkerConfig"], List[str]]
    options: Tuple[str, ...] = ()          # option names; all required, positive integers
    uses_overlap: bool = True              # False: ChunkerConfig.overlap must be 0
    check: Callable[["ChunkerConfig"], None] = lambda config: None


STRATEGIES: Dict[str, Strategy] = {
    "fixed": Strategy(_split_fixed),
    "recursive": Strategy(_split_recursive),
    "sentence": Strategy(_split_sentence),
    "sliding_window": Strategy(
        _split_sliding_window, options=("window", "stride"), uses_overlap=False, check=_check_sliding_window
    ),
}


def _validate_strategy(config: "ChunkerConfig") -> None:
    spec = STRATEGIES.get(config.strategy)
    if spec is None:
        raise ValueError(f"unknown chunking strategy {config.strategy!r}; available: {sorted(STRATEGIES)}")
    
    if set(config.options) != set(spec.options):
        raise ValueError(
            f"chunking strategy {config.strategy!r} takes options {list(spec.options)}, got {sorted(config.options)}"
        )
    for name, value in config.options.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"chunking option {name!r} must be a positive integer, got {value!r}")
    if not spec.uses_overlap and config.overlap != 0:
        raise ValueError(f"chunking strategy {config.strategy!r} does not use overlap; set overlap to 0")
    spec.check(config)


def split_text(text: str, config: "ChunkerConfig") -> List[str]:
    return STRATEGIES[config.strategy].split(text, config)


def chunk_passage(passage: Passage, config: ChunkerConfig)-> List[Chunk]:
    """Split one passage text into chunks with the configured strategy."""

    text = passage.text

    if not text:
        return []

    pieces = split_text(text, config)

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
