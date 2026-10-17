
"""
Chroma wrapper via langchain-chroma.

Internal to BuiltinRAGSystem only — not part of the public RAGSystem
contract. A user's own pipeline (see interface.py) retrieves however it
wants inside its own retrieve(); it never imports this module.

Uses the Chroma vectorstore's convenience API (add_texts / similarity_search)
against LangChainEmbeddingsAdapter.

"""
import hashlib
from dataclasses import asdict
from typing import List
 
from langchain_chroma import Chroma
 
from ragregress.rag.chunker import ChunkerConfig
from ragregress.rag.embedder import LangChainEmbeddingsAdapter
from ragregress.rag.interface import Chunk

def _chunker_figerprint(chunker_config: ChunkerConfig) -> str:
    """Hash of EVERY field on ChunkerConfig, not a hand-picked subset —
    so a new field added later (e.g. a future chunking `strategy`) is
    automatically covered without anyone needing to remember to update
    collection_name().

    Uses BLAKE2b with digest_size=8 (64 bits, 16 hex
    chars) natively rather than truncating a longer hash's hex string —
    digest_size is a first-class BLAKE2 parameter, not a post-hoc slice.

    """

    payload = repr(sorted(asdict(chunker_config).items())).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()

def content_fingerprint(chunks: List[Chunk]) -> str:
    """Hash of every chunk's id and text, order-independent"""
    h = hashlib.blake2b(digest_size=8)
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()

_FP_KEY = "content_fp"

def collection_name(embedding_model: str, chunker_config: ChunkerConfig) -> str:
    """passages_{model}_{chunk_size}_{fingerprint}

    Uniqueness is guaranteed by `fingerprint` alone (hash of the full
    chunker config) — `chunk_size` in the name is cosmetic, for humans
    skimming Chroma's storage, and plays no role in preventing collisions

    That split matters: it means two configs that differ only in
    `overlap` (or any future chunker field) still get distinct
    collections, even though their `chunk_size` looks the same in the name.

    """

    safe_model = embedding_model.split("/")[-1].replace(":","_")

    fingerprint = _chunker_figerprint(chunker_config)

    return f"passages_{safe_model}_{chunker_config.chunk_size}_{fingerprint}"


class ChromaRetriever:
    def __init__(
        self,
        embeddings_adapter: LangChainEmbeddingsAdapter,
        persist_directory: str,
        collection_name: str,
    ):
        self.collection_name = collection_name
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_adapter,
            persist_directory=persist_directory,
        )

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        metadatas = [
            {"chunk_id": c.chunk_id, "source_passage_id": c.source_passage_id, **c.metadata}
            for c in chunks
        ]

        self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def count(self) -> int:
            """Return number of chunks in this collection"""
    
            return len(self.store.get(include=[])['ids'])

    def _raw_collection(self):
        col = getattr(self.store, "_collection", None)
        if col is None:
            raise RuntimeError(
                "langchain-chroma no longer exposes Chroma._collection; "
                "update ChromaRetriever._raw_collection for this version."
            )
        return col

    def replace_all(self, chunks: List[Chunk], fingerprint: str) -> None:
        """Empty the collection, upsert `chunks`, then record `fingerprint`"""
        
        self.store.reset_collection()
        self.upsert(chunks)
        self._raw_collection().modify(metadata={_FP_KEY: fingerprint})
    

    def stored_fingerprint(self) -> str | None:
        """Content fingerprint recorded by replace_all(), or None if absent."""
        return (self._raw_collection().metadata or {}).get(_FP_KEY)

    def query(self, query_text: str, k: int = 5)-> List[Chunk]:

        results = self.store.similarity_search(query_text, k)

        out: List[Chunk] = []

        for doc in results:
            meta = dict(doc.metadata)
            source_passage_id = meta.pop("source_passage_id")
            chunk_id = meta.pop("chunk_id", "")

            out.append(
                Chunk(chunk_id= chunk_id,
                      source_passage_id= source_passage_id,
                      text= doc.page_content,
                      metadata = meta
                    )
            )

        return out

    

    

    