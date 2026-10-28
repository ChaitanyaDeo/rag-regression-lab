"""
Wires chunker -> embedder -> retriever behind the RAGSystem interface.

BuiltinRAGSystem is the lab's own reference/baseline RAGSystem
implementation — used for own dev work and as the thing
target-vs-actual metrics get measured against. It is not a base class or
required starting point for a user's pipeline: a user implements
RAGSystem directly (see interface.py) and is free to ignore chunker.py,
embedder.py, and retriever.py entirely.

"""


from typing import List, Optional
 
from ragregress.benchmark.schemas import Passage
from ragregress.rag.chunker import ChunkerConfig, chunk_passages
from ragregress.rag.embedder import DEFAULT_EMBEDDING_MODEL, Embedder, LangChainEmbeddingsAdapter
from ragregress.rag.interface import Chunk, GenerationResult
from ragregress.rag.reranker import Reranker
from ragregress.rag.retriever import ChromaRetriever, collection_name, content_fingerprint
from ragregress.rag.generator import GeneratorConfig, generate_with_citations


class BuiltinRAGSystem:
    def __init__(self, chunker_config: ChunkerConfig,
                 generator_config: GeneratorConfig = None,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 persist_directory: str = "artifacts/chroma",
                 reranker: Optional[Reranker] = None,
                 rerank_candidates: int = 0,
    ):
        if reranker is not None and rerank_candidates < 1:
            raise ValueError(f"rerank_candidates must be >= 1 when a reranker is set, got {rerank_candidates}")
        
        self.chunker_config = chunker_config
        self.embedding_model = embedding_model
        self.embedder = Embedder(embedding_model)
        self.embeddings_adapter = LangChainEmbeddingsAdapter(self.embedder)
        self.retriever = ChromaRetriever(
            embeddings_adapter=self.embeddings_adapter,
            persist_directory=persist_directory,
            collection_name=collection_name(embedding_model, chunker_config),
        )

        self.generator_config = generator_config or GeneratorConfig()
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates



    def ingest(self, passages: List[Passage]) -> int:
        """Chunk all passages and embed/store the resulting chunks in the retriever.

            Each passage is split using the configured chunk size and overlap via
            ``chunk_passages()``. The resulting flat list of chunks is then upserted
            through the retriever, which embeds the chunk text and stores the vectors
            in the Chroma collection associated with the embedding model and chunker
            configuration. Each stored chunk retains its ``chunk_id`` and
            ``source_passage_id`` metadata for traceability to the originating passage.
        """

        chunks = chunk_passages(passages, self.chunker_config)
        self.retriever.upsert(chunks)
        return len(chunks)

    def ensure_indexed(self, passages: List[Passage],  force: bool = False)-> int:
        """ 
            A content fingerprint (hash of all chunk ids + texts) is stored in the
            collection metadata. The collection is rebuilt from scratch when:
            - `force` is True (CLI --reindex), or
            - no fingerprint is stored (never indexed, partial ingest, or an
                index built before fingerprints existed), or
            - the fingerprint differs (chunking code or corpus changed), or
            - the stored count differs from the expected count.

            Returns the number of chunks embedded 
        """

        chunks = chunk_passages(passages, self.chunker_config)
        fp = content_fingerprint(chunks)

        if (not force and self.retriever.stored_fingerprint() == fp and self.retriever.count() == len(chunks)):
            return 0

        self.retriever.replace_all(chunks, fp)
        return len(chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        """Retrives the chunks for query from chromaDB

            It emmbeds the query and search for it inside DB
        """
        if self.reranker is None:
            return self.retriever.query(query, k=k)

        candidates = self.retriever.query(query, k = max(self.rerank_candidates, k))
        return self.reranker.rerank(query, candidates,k)

    def generate(self, query: str, contexts: List[Chunk]) -> GenerationResult:
        """generates the answer via (generator.py + ChatOllama wiring)

        Orchestration only — prompt-building, the LLM call, and citation
        parsing all live in generator.py so the same logic can be reused
        (e.g. by the no-RAG baseline generator).
        """

        return generate_with_citations(query, contexts, self.generator_config)
        
    

