"""
Text Chunker Module for DocuMind AI.
Divides document text into semantically coherent, overlapping chunks
using sentence-aware and paragraph-aware boundary detection.
"""

from typing import List, Dict, Any, Optional
import re


class TextChunker:
    """
    Chunks document content into overlapping windows with sentence-aware boundaries
    and complete metadata attribution (document name, page number, chunk ID).
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, processed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks all pages/blocks in a processed document.

        Args:
            processed_doc: Dictionary produced by DocumentProcessor.process_file().

        Returns:
            List of chunk dictionaries with metadata (chunk_id, document_name, page_number, text).
        """
        all_chunks: List[Dict[str, Any]] = []
        doc_name = processed_doc.get("document_name", "document")
        doc_id = processed_doc.get("doc_id", "doc_default")
        pages = processed_doc.get("pages", [])

        chunk_counter = 1

        for page in pages:
            page_num = page.get("page_number", None)
            page_text = page.get("text", "").strip()

            if not page_text:
                continue

            page_chunks = self._chunk_text(
                text=page_text,
                doc_name=doc_name,
                doc_id=doc_id,
                page_number=page_num,
                start_chunk_idx=chunk_counter
            )

            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

        return all_chunks

    def _chunk_text(
        self,
        text: str,
        doc_name: str,
        doc_id: str,
        page_number: Optional[int],
        start_chunk_idx: int
    ) -> List[Dict[str, Any]]:
        """Splits arbitrary text into overlapping chunks respecting sentence boundaries."""
        chunks: List[Dict[str, Any]] = []
        text_len = len(text)

        # If text fits into a single chunk
        if text_len <= self.chunk_size:
            clean_text = text.strip()
            if clean_text:
                chunks.append({
                    "chunk_id": f"{doc_id}_chk_{start_chunk_idx:04d}",
                    "document_name": doc_name,
                    "page_number": page_number,
                    "text": clean_text,
                    "char_count": len(clean_text)
                })
            return chunks

        start = 0
        current_idx = start_chunk_idx

        while start < text_len:
            end = start + self.chunk_size

            # Last slice
            if end >= text_len:
                chunk_slice = text[start:].strip()
                if chunk_slice:
                    chunks.append({
                        "chunk_id": f"{doc_id}_chk_{current_idx:04d}",
                        "document_name": doc_name,
                        "page_number": page_number,
                        "text": chunk_slice,
                        "char_count": len(chunk_slice)
                    })
                break

            # Find a natural breakpoint (paragraph -> sentence -> newline -> space)
            break_point = self._find_natural_break(text, start, end)
            chunk_slice = text[start:break_point].strip()

            if chunk_slice:
                chunks.append({
                    "chunk_id": f"{doc_id}_chk_{current_idx:04d}",
                    "document_name": doc_name,
                    "page_number": page_number,
                    "text": chunk_slice,
                    "char_count": len(chunk_slice)
                })
                current_idx += 1

            # Advance by step (respecting overlap)
            next_start = break_point - self.chunk_overlap
            if next_start <= start:
                start = break_point
            else:
                start = next_start

        return chunks

    def _find_natural_break(self, text: str, start: int, target_end: int) -> int:
        """Finds natural boundary (paragraph -> sentence -> newline -> word boundary)."""
        window_start = max(start + int(self.chunk_size * 0.5), target_end - 150)
        window = text[window_start:target_end]

        # 1. Paragraph boundary
        p_idx = window.rfind("\n\n")
        if p_idx != -1:
            return window_start + p_idx + 2

        # 2. Sentence boundary (. ! ?)
        sentence_matches = list(re.finditer(r'[.?!]\s+', window))
        if sentence_matches:
            return window_start + sentence_matches[-1].end()

        # 3. Newline
        nl_idx = window.rfind("\n")
        if nl_idx != -1:
            return window_start + nl_idx + 1

        # 4. Word boundary
        space_idx = window.rfind(" ")
        if space_idx != -1:
            return window_start + space_idx + 1

        return target_end
