"""
Prompt Engineering & Hallucination Prevention Module for DocuMind AI.
Enforces strict factual grounding, refusal handling for missing context,
and natural language answer formulation.
"""

from typing import List, Dict, Any, Optional

# System prompt mandated for DocuMind AI
DOCUMIND_SYSTEM_PROMPT = """You are DocuMind AI, an intelligent document assistant.

Your responsibility is to answer questions based on the information retrieved from the user's uploaded document.

You must follow these rules:

1. Understand the user's question, even if it is phrased differently from the original document.
2. Use the supplied document context as the factual basis of your answer.
3. Do not invent facts, figures, dates, names or statistics.
4. Do not use outside knowledge to fill missing information.
5. If the answer cannot be found in the supplied context, clearly state that the document does not contain sufficient information to answer the question, and politely decline to speculate.
6. Provide clear, direct, and well-structured answers."""

STANDARD_REFUSAL_PHRASE = "The uploaded document does not contain sufficient information to answer this question."


def build_user_prompt(question: str, retrieved_context: str) -> str:
    """
    Constructs the user message incorporating retrieved document context
    and the user's question.
    """
    return f"""Retrieved Document Context:
=========================================
{retrieved_context}
=========================================

User Question: {question}

Please answer the question accurately and concisely, based strictly on the retrieved document context above."""


def is_refusal_response(response_text: str) -> bool:
    """
    Detects whether the LLM indicated that the document lacks sufficient information.
    """
    lower = response_text.lower().strip()
    refusal_markers = [
        "not contain sufficient information",
        "does not contain sufficient information",
        "insufficient information",
        "could not find sufficient information",
        "cannot find sufficient information",
        "not mentioned in the uploaded",
        "not mentioned in the provided",
        "does not mention",
        "no information provided",
        "not found in the document",
        "document does not contain",
        "context does not contain",
        "cannot provide an answer",
        "does not contain any information",
        "not contain any information",
        "no mention of"
    ]
    return any(marker in lower for marker in refusal_markers)
