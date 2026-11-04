"""
FAISS Vector Retrieval Store for DocuMind AI.
Indexes chunk embeddings using faiss.IndexFlatIP for exact Cosine Similarity.
Maintains 1:1 mapping between FAISS vector IDs and rich chunk metadata.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss


class FAISSVectorStore:
    """
    Local FAISS CPU vector index storing chunk embeddings and metadata.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds chunk metadata and embeddings to the FAISS index.

        Args:
            chunks: List of chunk metadata dictionaries.
            embeddings: Float32 numpy array of shape (N, dimension).
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}.")

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K nearest chunks based on Cosine Similarity (inner product of unit vectors).

        Args:
            query_vector: 2D numpy array of shape (1, dimension).
            top_k: Number of chunks to retrieve.

        Returns:
            Ranked list of matching chunk dictionaries with 'similarity_score'.
        """
        if self.is_empty():
            return []

        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, actual_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk_data = dict(self.metadata[idx])
            chunk_data["similarity_score"] = float(score)
            chunk_data["rank"] = rank
            results.append(chunk_data)

        return results

    def clear(self):
        """Resets the vector index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata.clear()

    @property
    def total_vectors(self) -> int:
        """Total vectors currently indexed."""
        return self.index.ntotal if self.index is not None else 0

    def is_empty(self) -> bool:
        """True if no chunks are currently indexed."""
        return self.total_vectors == 0
