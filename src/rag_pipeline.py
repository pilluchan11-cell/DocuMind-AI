"""
RAG Pipeline Module for DocuMind AI.
Orchestrates end-to-end question answering:
1. Semantic search in FAISS using SentenceTransformers embeddings.
2. Context assembly with transparent source citations.
3. Local inference via Ollama /api/chat with strict grounding and hallucination prevention.
"""

from typing import Dict, Any, List, Optional, Generator, Tuple
from src.retriever import Retriever
from src.ollama_client import OllamaClient, OllamaClientError
from src.prompt_templates import (
    DOCUMIND_SYSTEM_PROMPT,
    build_user_prompt,
    is_refusal_response
)


class RAGPipeline:
    """
    Coordinates Retrieval-Augmented Generation for single-document Q&A.
    """

    def __init__(self, retriever: Retriever, ollama_client: OllamaClient):
        self.retriever = retriever
        self.ollama_client = ollama_client

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 400
    ) -> Dict[str, Any]:
        """
        Executes synchronous end-to-end question answering.

        Args:
            query: The user question.
            top_k: Number of relevant chunks to retrieve.
            model_name: Optional Ollama model override.
            temperature: Sampling temperature (0.1 for high factual grounding).
            chat_history: Prior conversational messages.
            max_tokens: Max generation tokens.

        Returns:
            Dictionary with question, answer, sources, and refusal status.
        """
        if not query or not query.strip():
            return {
                "question": query,
                "answer": "Please ask a question about your uploaded document.",
                "sources": [],
                "is_refusal": False,
                "error": None
            }

        # 1. Expand query with prior conversational context if pronoun-heavy or very short
        search_query = self._build_search_query(query, chat_history)

        # 2. Retrieve top-K relevant chunks from FAISS
        retrieval_result = self.retriever.retrieve(query=search_query, top_k=top_k)
        retrieved_chunks = retrieval_result.get("chunks", [])
        context_text = retrieval_result.get("context_text", "")

        if not retrieved_chunks:
            return {
                "question": query,
                "answer": "No document has been uploaded yet. Please upload a document above to begin.",
                "sources": [],
                "is_refusal": True,
                "error": None
            }

        # 3. Assemble chat messages for Ollama /api/chat
        messages = self._build_chat_messages(
            current_question=query,
            context_text=context_text,
            chat_history=chat_history
        )

        # 4. Local Ollama inference via /api/chat
        active_model = model_name or self.ollama_client.default_model
        try:
            raw_answer = self.ollama_client.chat(
                messages=messages,
                model=active_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer_text = raw_answer.strip()
            error_msg = None
        except OllamaClientError as e:
            answer_text = f"**Ollama Generation Error**: {str(e)}"
            error_msg = str(e)

        # 5. Hallucination check
        refusal = is_refusal_response(answer_text)

        # 6. Format source citations
        formatted_sources = self._format_sources(retrieved_chunks)

        return {
            "question": query,
            "answer": answer_text,
            "sources": formatted_sources,
            "is_refusal": refusal,
            "model_used": active_model,
            "error": error_msg
        }

    def answer_query_stream(
        self,
        query: str,
        top_k: int = 5,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 400
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Streams token chunks iteratively from Ollama /api/chat while returning retrieved sources.

        Returns:
            Tuple of (token_generator, sources_list)
        """
        search_query = self._build_search_query(query, chat_history)
        retrieval_result = self.retriever.retrieve(query=search_query, top_k=top_k)
        retrieved_chunks = retrieval_result.get("chunks", [])
        context_text = retrieval_result.get("context_text", "")

        formatted_sources = self._format_sources(retrieved_chunks)

        if not retrieved_chunks:
            def empty_gen():
                yield "No document has been uploaded yet. Please upload a document to begin."
            return empty_gen(), []

        messages = self._build_chat_messages(
            current_question=query,
            context_text=context_text,
            chat_history=chat_history
        )

        active_model = model_name or self.ollama_client.default_model
        token_stream = self.ollama_client.chat_stream(
            messages=messages,
            model=active_model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return token_stream, formatted_sources

    @staticmethod
    def _build_search_query(query: str, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Enriches retrieval search query with previous conversation turn if referring to prior context."""
        if not chat_history:
            return query

        q_lower = query.lower()
        pronouns = ["it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "she", "him", "her"]
        words = q_lower.split()

        # If query is short or contains pronouns, include the last user question for search
        if any(p in words for p in pronouns) or len(words) <= 4:
            user_msgs = [m.get("content", "") for m in chat_history if m.get("role") == "user"]
            if user_msgs:
                return f"{user_msgs[-1]} {query}"

        return query

    @staticmethod
    def _build_chat_messages(
        current_question: str,
        context_text: str,
        chat_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Constructs the messages payload for Ollama /api/chat."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": DOCUMIND_SYSTEM_PROMPT}
        ]

        # Append recent conversational context (up to last 2 complete turns)
        if chat_history:
            # Filter valid user/assistant turns
            valid_turns = [m for m in chat_history if m.get("role") in ("user", "assistant") and m.get("content")]
            # Keep up to last 4 messages (2 turns)
            recent_turns = valid_turns[-4:]
            for turn in recent_turns:
                # Omit full document context from past turns to save context window
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })

        # Append the current turn with fresh retrieved context
        user_prompt = build_user_prompt(question=current_question, retrieved_context=context_text)
        messages.append({"role": "user", "content": user_prompt})

        return messages

    @staticmethod
    def _format_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Formats chunk citations with document name, page number, and original text."""
        formatted = []
        for chk in retrieved_chunks:
            formatted.append({
                "chunk_id": chk.get("chunk_id", "N/A"),
                "document_name": chk.get("document_name", "Document"),
                "page_number": chk.get("page_number", None),
                "similarity_score": round(chk.get("similarity_score", 0.0), 4),
                "text": chk.get("text", "").strip()
            })
        return formatted
