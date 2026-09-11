
"""
Hand-rolled sentence-transformers embedder, plus a thin adapter so it
composes with langchain-chroma's vectorstore convenience API.

Internal to BuiltinRAGSystem only — not part of the public RAGSystem
contract. A user's own pipeline (see interface.py) embeds however it
wants inside its own retrieve(); it never imports this module


embedder.py stays a plain
sentence-transformers wrapper, not a LangChain Embeddings subclass. The
adapter below is the seam — swap the embedding backend by changing
Embedder only, and retriever.py never needs to know sentence-transformers
exists.
"""

from typing import List, Optional
 
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



class Embedder:
    def __init__(
            self,
            model_name: str = DEFAULT_EMBEDDING_MODEL,
            device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    )-> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return vectors.tolist()


class LangChainEmbeddingsAdapter:
    """Duck-types langchain_core.embeddings.Embeddings (embed_documents /
    embed_query) so a Chroma() vectorstore instance accepts it directly as
    embedding_function, without embedder.py depending on langchain-core.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.encode([text])[0]