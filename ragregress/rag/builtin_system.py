"""
Wires chunker -> embedder -> retriever behind the RAGSystem interface.

BuiltinRAGSystem is the lab's own reference/baseline RAGSystem
implementation — used for own dev work and as the thing
target-vs-actual metrics get measured against. It is not a base class or
required starting point for a user's pipeline: a user implements
RAGSystem directly (see interface.py) and is free to ignore chunker.py,
embedder.py, and retriever.py entirely.

"""


from typing import List
 
from ragregress.benchmark.schemas import Passage
from ragregress.rag.chunker import ChunkerConfig, chunk_passages
from ragregress.rag.embedder import DEFAULT_EMBEDDING_MODEL, Embedder, LangChainEmbeddingsAdapter
from ragregress.rag.interface import Chunk
from ragregress.rag.retriever import ChromaRetriever, collection_name


class BuiltinRAGSystem:
    def __init__(self, chunker_config: ChunkerConfig, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "artifacts/chroma",
    ):
        self.chunker_config = chunker_config
        self.embedding_model = embedding_model
        self.embedder = Embedder(embedding_model)
        self.embeddings_adapter = LangChainEmbeddingsAdapter(self.embedder)
        self.retriever = ChromaRetriever(
            embeddings_adapter=self.embeddings_adapter,
            persist_directory=persist_directory,
            collection_name=collection_name(embedding_model, chunker_config),
        )



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

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        """Retrives the chunks for query from chromaDB

            It emmbeds the query and search for it inside DB
        """
        return self.retriever.query(query, k=k)

    def generate(self, query: str, contexts: List[Chunk]) -> str:
        """generates the answer via (generator.py + ChatOllama wiring)

        TODO: implement generator
        """
        raise NotImplementedError("generator is not implemented yet")
    

