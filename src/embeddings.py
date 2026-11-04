"""
Embedding Generation Module for DocuMind AI.
Generates 384-dimensional dense semantic vectors using Sentence Transformers (all-MiniLM-L6-v2).
Embeddings are L2-normalized so that inner product equals exact cosine similarity.
Runs entirely locally on CPU with zero paid cloud APIs.
"""

from typing import List, Any
import numpy as np


class EmbeddingManager:
    """
    Loads and reuses the SentenceTransformer all-MiniLM-L6-v2 model locally.
    Produces L2-normalized embeddings for fast cosine similarity search in FAISS.
    """

    _instance = None
    _model = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384
        self._load_model()

    def _load_model(self):
        """Loads the SentenceTransformer model once and caches it in the class."""
        if EmbeddingManager._model is None:
            from sentence_transformers import SentenceTransformer
            EmbeddingManager._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates L2-normalized float32 embeddings for a list of text strings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for model inference.

        Returns:
            np.ndarray of shape (len(texts), 384) with float32 values.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()
        embeddings = EmbeddingManager._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: List[Any], batch_size: int = 32) -> np.ndarray:
        """
        Generates L2-normalized embeddings for a list of chunk dictionaries.
        """
        texts = [c["text"] if isinstance(c, dict) and "text" in c else str(c) for c in chunks]
        return self.embed_texts(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates L2-normalized embedding for a single user query.

        Args:
            query: The user query string.

        Returns:
            np.ndarray of shape (1, 384) with float32 values.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        return self.embed_texts([query.strip()], batch_size=1)

    def get_dimension(self) -> int:
        """Returns embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        return self.dimension
