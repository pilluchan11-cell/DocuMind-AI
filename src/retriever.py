"""
Retriever Module for DocuMind AI.
Connects query embedding with FAISS similarity search to retrieve the most relevant
document chunks and format them into grounded context blocks for the LLM.
"""

from typing import List, Dict, Any
from src.embeddings import EmbeddingManager
from src.vector_store import FAISSVectorStore


class Retriever:
    """
    Retrieves the most semantically relevant text chunks for a user query.
    """

    def __init__(self, embedding_manager: EmbeddingManager, vector_store: FAISSVectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieves top-K relevant chunks for the provided question.

        Args:
            query: The user question.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            Dictionary with query, chunks list, formatted context_text, and count.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "chunks": [],
                "context_text": "",
                "total_retrieved": 0
            }

        if self.vector_store.is_empty():
            return {
                "query": query,
                "chunks": [],
                "context_text": "[No document indexed in knowledge base]",
                "total_retrieved": 0
            }

        # 1. Generate L2-normalized query vector using SentenceTransformer
        query_vector = self.embedding_manager.embed_query(query)

        # 2. Search FAISS index using Cosine Similarity
        chunks = self.vector_store.search(query_vector=query_vector, top_k=top_k)

        # 3. Format into clear LLM context blocks
        context_text = self.build_context_block(chunks)

        return {
            "query": query,
            "chunks": chunks,
            "context_text": context_text,
            "total_retrieved": len(chunks)
        }

    @staticmethod
    def build_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Builds a clean, readable context string from retrieved chunks.
        """
        if not retrieved_chunks:
            return "[No relevant document context found]"

        blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            doc_name = chunk.get("document_name", "Document")
            page_num = chunk.get("page_number")
            page_label = f"Page {page_num}" if page_num is not None else "Page N/A"
            text = chunk.get("text", "").strip()

            blocks.append(
                f"[Source {idx} | {doc_name} | {page_label}]\n{text}"
            )

        return "\n\n".join(blocks)
