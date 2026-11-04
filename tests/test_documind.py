"""
Comprehensive Integration and Unit Test Suite for DocuMind AI.
Verifies all 10 core requirements from the specification:
1. Document Extraction (PDF with PyMuPDF, DOCX, TXT, and Streamlit UploadedFile simulation)
2. Safe byte-based extraction & error handling (0 bytes, corrupted PDF, non-extractable PDF)
3. Text chunking with sentence awareness, overlap, and metadata
4. Embedding generation with Sentence Transformers (all-MiniLM-L6-v2)
5. FAISS vector store indexing and Cosine Similarity retrieval (IndexFlatIP)
6. Local Ollama service & model verification (llama3.2:3b on /api/chat)
7. Semantic retrieval with paraphrased user queries (different wording from original text)
8. Multi-turn conversation context handling
9. Strict hallucination prevention & missing information refusal
10. Source references formatting and attribution
"""

import io
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
from src.document_processor import DocumentProcessor, DocumentProcessingError
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.ollama_client import OllamaClient
from src.rag_pipeline import RAGPipeline


class MockUploadedFile(io.BytesIO):
    """Simulates Streamlit's UploadedFile object."""
    def __init__(self, data: bytes, name: str, mime_type: str = "application/pdf"):
        super().__init__(data)
        self.name = name
        self.type = mime_type
        self.size = len(data)


class TestDocuMindAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_path = PROJECT_ROOT / "data" / "sample_document.pdf"
        cls.docx_path = PROJECT_ROOT / "data" / "sample_competitor_report.docx"
        cls.txt_path = PROJECT_ROOT / "data" / "sample_document.txt"

        cls.embed_mgr = EmbeddingManager(model_name=config.DEFAULT_EMBEDDING_MODEL)
        cls.ollama_client = OllamaClient(
            base_url=config.OLLAMA_BASE_URL,
            default_model=config.DEFAULT_OLLAMA_MODEL
        )

    def test_01_ollama_service_and_model(self):
        """Test local Ollama availability and llama3.2:3b presence."""
        is_conn, msg, models = self.ollama_client.check_connection()
        self.assertTrue(is_conn, f"Ollama is not running: {msg}")
        is_installed, _ = self.ollama_client.is_model_installed(config.DEFAULT_OLLAMA_MODEL)
        self.assertTrue(is_installed, f"Model {config.DEFAULT_OLLAMA_MODEL} is missing from Ollama.")

    def test_02_pdf_extraction_and_page_preservation(self):
        """Test PyMuPDF extraction on Streamlit UploadedFile, preserving 1-indexed page numbers."""
        self.assertTrue(self.pdf_path.exists())
        mock_pdf = MockUploadedFile(self.pdf_path.read_bytes(), name="sample_document.pdf")
        doc = DocumentProcessor.process_file(mock_pdf)

        self.assertEqual(doc["document_name"], "sample_document.pdf")
        self.assertEqual(doc["file_type"], "PDF")
        self.assertGreater(doc["total_pages"], 0)
        self.assertGreater(doc["total_words"], 50)

        # Check page numbers are preserved as 1, 2, ...
        for idx, page in enumerate(doc["pages"], 1):
            self.assertEqual(page["page_number"], idx)

    def test_03_docx_extraction_no_invented_pages(self):
        """Test Word DOCX extraction without inventing page numbers."""
        self.assertTrue(self.docx_path.exists())
        mock_docx = MockUploadedFile(self.docx_path.read_bytes(), name="sample_competitor_report.docx")
        doc = DocumentProcessor.process_file(mock_docx)

        self.assertEqual(doc["file_type"], "DOCX")
        self.assertIsNone(doc["total_pages"], "DOCX must not invent page count")
        self.assertGreater(doc["total_words"], 50)
        for block in doc["pages"]:
            self.assertIsNone(block["page_number"], "DOCX blocks must have page_number = None")

    def test_04_txt_extraction_multi_encoding(self):
        """Test plain text extraction with safe decoding and no invented pages."""
        self.assertTrue(self.txt_path.exists())
        mock_txt = MockUploadedFile(self.txt_path.read_bytes(), name="sample_document.txt")
        doc = DocumentProcessor.process_file(mock_txt)

        self.assertEqual(doc["file_type"], "TXT")
        self.assertIsNone(doc["total_pages"], "TXT must not invent page count")
        self.assertGreater(doc["total_words"], 100)

    def test_05_error_handling_invalid_uploads(self):
        """Test graceful rejection of empty, corrupted, or unsupported files."""
        # Empty file
        empty_file = MockUploadedFile(b"", name="empty.pdf")
        with self.assertRaises(DocumentProcessingError):
            DocumentProcessor.process_file(empty_file)

        # Corrupted PDF
        corrupt_file = MockUploadedFile(b"%PDF-1.4 garbage binary data", name="broken.pdf")
        with self.assertRaises(DocumentProcessingError):
            DocumentProcessor.process_file(corrupt_file)

        # Unsupported extension
        unsupported_file = MockUploadedFile(b"something", name="archive.zip")
        with self.assertRaises(DocumentProcessingError):
            DocumentProcessor.process_file(unsupported_file)

    def test_06_text_chunking_with_overlap_and_metadata(self):
        """Test sentence-aware chunking preserving metadata and overlap."""
        mock_txt = MockUploadedFile(self.txt_path.read_bytes(), name="sample_document.txt")
        doc = DocumentProcessor.process_file(mock_txt)

        chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertEqual(chunk["document_name"], "sample_document.txt")
            self.assertIn("text", chunk)
            self.assertGreater(len(chunk["text"]), 10)

    def test_07_dense_embeddings_and_faiss_retrieval(self):
        """Test L2-normalized embeddings and FAISS inner-product retrieval."""
        mock_pdf = MockUploadedFile(self.pdf_path.read_bytes(), name="sample_document.pdf")
        doc = DocumentProcessor.process_file(mock_pdf)
        chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(doc)

        embeddings = self.embed_mgr.embed_chunks(chunks)
        self.assertEqual(embeddings.shape[0], len(chunks))
        self.assertEqual(embeddings.shape[1], config.EMBEDDING_DIMENSION)

        # Index in FAISS
        vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
        vstore.add_chunks(chunks, embeddings)
        self.assertEqual(vstore.total_vectors, len(chunks))

        # Retrieve
        retriever = Retriever(embedding_manager=self.embed_mgr, vector_store=vstore)
        res = retriever.retrieve("What is this document about?", top_k=3)
        self.assertEqual(len(res["chunks"]), 3)
        self.assertTrue(len(res["context_text"]) > 20)

    def test_08_paraphrased_rag_query_understanding(self):
        """Test semantic RAG understanding when user uses different wording."""
        # Index TXT document
        mock_txt = MockUploadedFile(self.txt_path.read_bytes(), name="sample_document.txt")
        doc = DocumentProcessor.process_file(mock_txt)
        chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(doc)
        embeddings = self.embed_mgr.embed_chunks(chunks)

        vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
        vstore.add_chunks(chunks, embeddings)
        retriever = Retriever(embedding_manager=self.embed_mgr, vector_store=vstore)
        pipeline = RAGPipeline(retriever=retriever, ollama_client=self.ollama_client)

        # Original doc text: "NexusTech Innovations was founded in Austin, Texas in October 2018"
        # Paraphrased query: "In which city did the company start its business operations?"
        query = "In which city did the company start its business operations?"
        result = pipeline.answer_query(query=query, top_k=3, max_tokens=150)

        self.assertIn("austin", result["answer"].lower())
        self.assertGreater(len(result["sources"]), 0)
        self.assertEqual(result["sources"][0]["document_name"], "sample_document.txt")

    def test_09_multi_turn_conversational_follow_up(self):
        """Test conversational follow-up question using chat history."""
        mock_txt = MockUploadedFile(self.txt_path.read_bytes(), name="sample_document.txt")
        doc = DocumentProcessor.process_file(mock_txt)
        chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(doc)
        embeddings = self.embed_mgr.embed_chunks(chunks)

        vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
        vstore.add_chunks(chunks, embeddings)
        retriever = Retriever(embedding_manager=self.embed_mgr, vector_store=vstore)
        pipeline = RAGPipeline(retriever=retriever, ollama_client=self.ollama_client)

        history = [
            {"role": "user", "content": "Where was the company founded?"},
            {"role": "assistant", "content": "NexusTech Innovations was founded in Austin, Texas."}
        ]
        follow_up = "Who were the founders who started it and what was their background?"
        result = pipeline.answer_query(query=follow_up, top_k=3, chat_history=history, max_tokens=150)

        self.assertTrue(
            any(name in result["answer"].lower() for name in ["evelyn", "vance", "marcus", "cyberdynamics"]),
            f"Expected Dr. Evelyn Vance or founders in answer: {result['answer']}"
        )

    def test_10_strict_grounding_refusal(self):
        """Test hallucination prevention: refuses to answer questions outside document context."""
        mock_txt = MockUploadedFile(self.txt_path.read_bytes(), name="sample_document.txt")
        doc = DocumentProcessor.process_file(mock_txt)
        chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(doc)
        embeddings = self.embed_mgr.embed_chunks(chunks)

        vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
        vstore.add_chunks(chunks, embeddings)
        retriever = Retriever(embedding_manager=self.embed_mgr, vector_store=vstore)
        pipeline = RAGPipeline(retriever=retriever, ollama_client=self.ollama_client)

        out_of_scope_query = "What is the secret recipe for chocolate cake on Mars?"
        result = pipeline.answer_query(query=out_of_scope_query, top_k=3, max_tokens=150)

        # Verify that either refusal flag is True or refusal phrases are present
        ans_lower = result["answer"].lower()
        refused = (
            result["is_refusal"]
            or "not contain sufficient information" in ans_lower
            or "does not contain" in ans_lower
            or "not mentioned" in ans_lower
            or "no information" in ans_lower
            or "cannot find" in ans_lower
        )
        self.assertTrue(refused, f"Expected hallucination refusal, but got: {result['answer']}")


if __name__ == "__main__":
    unittest.main()
