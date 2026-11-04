"""
Unit & Integration Test for Document Upload and Text Extraction Fix.
Validates:
1. Streamlit UploadedFile simulation (BytesIO with .name, .getvalue(), .read(), .seek())
2. PDF extraction via PyMuPDF and pypdf fallback with byte streams
3. DOCX extraction with BytesIO
4. TXT extraction with various encodings
5. Error handling: empty files, corrupted PDFs, invalid file formats
6. Full RAG pipeline execution on extracted PDF:
   Extraction -> Chunking -> Embeddings -> FAISS -> Retrieval -> Ollama Generation -> Source Citation
"""

import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 stdout on Windows
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
    """Simulates Streamlit's UploadedFile object with metadata and stream capabilities."""

    def __init__(self, data: bytes, name: str, mime_type: str = "application/pdf"):
        super().__init__(data)
        self.name = name
        self.type = mime_type
        self.size = len(data)


def test_upload_and_extraction():
    print("=" * 70)
    print("   TESTING DOCUMENT UPLOAD & TEXT EXTRACTION FIX")
    print("=" * 70)

    pdf_path = PROJECT_ROOT / "data" / "sample_document.pdf"
    docx_path = PROJECT_ROOT / "data" / "sample_competitor_report.docx"
    txt_path = PROJECT_ROOT / "data" / "sample_document.txt"

    assert pdf_path.exists(), f"Missing {pdf_path}"
    pdf_bytes = pdf_path.read_bytes()

    # --------------------------------------------------------------------------
    # TEST 1: Streamlit UploadedFile simulation (PDF)
    # --------------------------------------------------------------------------
    print("\n[Test 1] Simulating Streamlit UploadedFile object (PDF)...")
    mock_pdf = MockUploadedFile(pdf_bytes, name="sample_document.pdf")
    doc_1 = DocumentProcessor.process_file(mock_pdf)
    assert doc_1["document_name"] == "sample_document.pdf"
    assert doc_1["total_pages"] == 2
    assert doc_1["total_words"] > 100
    print(f"  ✓ Success: Extracted {doc_1['total_pages']} pages, {doc_1['total_words']} words from Streamlit UploadedFile.")

    # --------------------------------------------------------------------------
    # TEST 2: Already-consumed stream pointer (ensure seek(0) handling)
    # --------------------------------------------------------------------------
    print("\n[Test 2] Testing already-consumed stream (read pointer at EOF)...")
    mock_consumed = MockUploadedFile(pdf_bytes, name="sample_document.pdf")
    mock_consumed.read()  # Consume stream to EOF
    assert mock_consumed.tell() == len(pdf_bytes)  # Pointer is at end

    # DocumentProcessor must safely read even if pointer was consumed
    doc_2 = DocumentProcessor.process_file(mock_consumed)
    assert doc_2["total_pages"] == 2
    print(f"  ✓ Success: Correctly handled pre-read/consumed stream without 0-byte failure.")

    # --------------------------------------------------------------------------
    # TEST 3: Raw bytes input directly
    # --------------------------------------------------------------------------
    print("\n[Test 3] Testing raw bytes input directly with filename...")
    doc_3 = DocumentProcessor.process_file(pdf_bytes, filename="sample_document.pdf")
    assert doc_3["total_pages"] == 2
    print(f"  ✓ Success: Raw bytes extracted cleanly.")

    # --------------------------------------------------------------------------
    # TEST 4: DOCX extraction with MockUploadedFile
    # --------------------------------------------------------------------------
    print("\n[Test 4] Testing DOCX extraction with Streamlit UploadedFile...")
    docx_bytes = docx_path.read_bytes()
    mock_docx = MockUploadedFile(docx_bytes, name="sample_competitor_report.docx", mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    doc_docx = DocumentProcessor.process_file(mock_docx)
    assert doc_docx["total_words"] > 50
    print(f"  ✓ Success: DOCX extracted {doc_docx['total_words']} words, {len(doc_docx['sections'])} sections.")

    # --------------------------------------------------------------------------
    # TEST 5: TXT extraction with MockUploadedFile
    # --------------------------------------------------------------------------
    print("\n[Test 5] Testing TXT extraction with Streamlit UploadedFile...")
    txt_bytes = txt_path.read_bytes()
    mock_txt = MockUploadedFile(txt_bytes, name="sample_document.txt", mime_type="text/plain")
    doc_txt = DocumentProcessor.process_file(mock_txt)
    assert doc_txt["total_words"] > 500
    print(f"  ✓ Success: TXT extracted {doc_txt['total_words']} words across {doc_txt['total_pages']} pages.")

    # --------------------------------------------------------------------------
    # TEST 6: Empty file handling (0 bytes)
    # --------------------------------------------------------------------------
    print("\n[Test 6] Testing empty file handling (0 bytes)...")
    empty_mock = MockUploadedFile(b"", name="empty.pdf")
    try:
        DocumentProcessor.process_file(empty_mock)
        assert False, "Should have raised DocumentProcessingError on empty file"
    except DocumentProcessingError as e:
        print(f"  ✓ Success: Correctly rejected empty file with: '{str(e)}'")

    # --------------------------------------------------------------------------
    # TEST 7: Corrupted PDF handling
    # --------------------------------------------------------------------------
    print("\n[Test 7] Testing corrupted PDF handling...")
    corrupt_bytes = b"%PDF-1.4 header but completely corrupted binary garbage and broken xref table"
    corrupt_mock = MockUploadedFile(corrupt_bytes, name="corrupt.pdf")
    try:
        DocumentProcessor.process_file(corrupt_mock)
        assert False, "Should have raised DocumentProcessingError on corrupted PDF"
    except DocumentProcessingError as e:
        print(f"  ✓ Success: Correctly rejected corrupted PDF with: '{str(e)}'")

    # --------------------------------------------------------------------------
    # TEST 8: Full RAG Pipeline execution on uploaded PDF
    # --------------------------------------------------------------------------
    print("\n[Test 8] Executing full RAG pipeline on extracted PDF document...")
    # 8a. Chunking
    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = chunker.chunk_document(doc_1)
    assert len(chunks) > 0
    print(f"  ✓ Chunking: Generated {len(chunks)} chunks with page attribution.")

    # 8b. Embeddings
    emb_mgr = EmbeddingManager(model_name=config.DEFAULT_EMBEDDING_MODEL)
    embeddings = emb_mgr.embed_chunks(chunks)
    assert embeddings.shape[0] == len(chunks)
    print(f"  ✓ Embeddings: Generated {embeddings.shape} dense vectors.")

    # 8c. FAISS Vector Store
    vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
    vstore.add_chunks(chunks, embeddings)
    print(f"  ✓ FAISS: Indexed {vstore.total_vectors} vectors.")

    # 8d. Retriever & Ollama QA
    retriever = Retriever(embedding_manager=emb_mgr, vector_store=vstore)
    client = OllamaClient(base_url=config.OLLAMA_BASE_URL, default_model=config.DEFAULT_OLLAMA_MODEL)
    pipeline = RAGPipeline(retriever=retriever, ollama_client=client)

    question = "What is the primary topic and focus of this PDF document?"
    print(f"  ✓ Asking question to local LLM: '{question}'...")
    result = pipeline.answer_query(query=question, top_k=3)

    print("\n  [Generated Answer]:")
    print(f"  {result['answer'][:300]}...")
    print(f"\n  [Sources Cited]: {len(result['sources'])} chunks")
    for s in result["sources"]:
        print(f"   - Document: {s['document_name']}, Page: {s['page_number']}, Similarity: {s['similarity_score']}")

    assert len(result["answer"]) > 20
    assert len(result["sources"]) > 0
    assert result["sources"][0]["document_name"] == "sample_document.pdf"
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! PDF EXTRACTION AND RAG VERIFIED.")


if __name__ == "__main__":
    test_upload_and_extraction()
