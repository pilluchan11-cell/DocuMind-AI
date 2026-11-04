"""
DocuMind AI - Verification of the Simple Single-Page RAG Workflow
Tests:
1. Document extraction (TXT, DOCX, PDF)
2. Chunk creation
3. Dense vector embeddings (all-MiniLM-L6-v2)
4. FAISS index creation and similarity retrieval
5. Ollama question understanding with paraphrased queries
6. Natural-language answer generation with strict document grounding
7. Follow-up question handling with conversational context
8. Source references verification
9. Unanswerable question refusal
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
from src.document_processor import DocumentProcessor
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.ollama_client import OllamaClient
from src.rag_pipeline import RAGPipeline


def test_full_workflow():
    print("=" * 70)
    print("   DOCUMIND AI - SIMPLE RAG WORKFLOW VERIFICATION TEST")
    print("=" * 70)

    # 1. Check Ollama
    client = OllamaClient(base_url=config.OLLAMA_BASE_URL, default_model=config.DEFAULT_OLLAMA_MODEL)
    conn, msg, models = client.check_connection()
    assert conn, f"Ollama not running: {msg}"
    print(f"[1/6] Ollama connection: OK (model: {config.DEFAULT_OLLAMA_MODEL})")

    # 2. Extract TXT document
    txt_path = PROJECT_ROOT / "data" / "sample_document.txt"
    assert txt_path.exists(), "sample_document.txt not found"
    with open(txt_path, "rb") as f:
        file_bytes = f.read()
    
    doc = DocumentProcessor.process_file(file_bytes, filename="sample_document.txt")
    print(f"[2/6] Extraction: OK (Pages: {doc['total_pages']}, Chars: {doc['total_chars']}, Words: {doc['total_words']})")

    # 3. Chunk text
    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 0, "No chunks generated"
    print(f"[3/6] Chunk creation: OK ({len(chunks)} chunks generated)")

    # 4. Embeddings & FAISS
    embed_mgr = EmbeddingManager(model_name=config.DEFAULT_EMBEDDING_MODEL)
    embeddings = embed_mgr.embed_chunks(chunks)
    assert embeddings.shape[0] == len(chunks), "Embedding count mismatch"

    vstore = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
    vstore.add_chunks(chunks, embeddings)
    assert not vstore.is_empty(), "FAISS store is empty"
    print(f"[4/6] Embeddings & FAISS storage: OK ({embeddings.shape[0]} vectors indexed)")

    # 5. Pipeline initialization
    retriever = Retriever(embedding_manager=embed_mgr, vector_store=vstore)
    pipeline = RAGPipeline(retriever=retriever, ollama_client=client)

    # 6. Test Paraphrased Question (Different wording from document)
    # Doc text: "NexusTech Innovations was founded in Austin, Texas in October 2018"
    # Query: "In which city did the company start its business operations?"
    print("\n--- Test A: Paraphrased Query Understanding ---")
    query_a = "In which city did the company start its business operations?"
    res_a = pipeline.answer_query(query=query_a, top_k=3)
    print(f"Query: {query_a}")
    print(f"Answer:\n{res_a['answer']}\n")
    print(f"Sources retrieved: {len(res_a['sources'])}")
    for s in res_a['sources']:
        print(f" - [{s['document_name']}] Page {s['page_number']}: {s['text'][:80]}...")
    assert "austin" in res_a["answer"].lower(), "Expected Austin in answer"
    assert len(res_a["sources"]) > 0, "Expected source citations"
    print(">>> PASS: Paraphrased query answered correctly with sources.")

    # 7. Test Follow-Up Question with Chat History
    print("\n--- Test B: Contextual Follow-Up Question ---")
    chat_history = [
        {"role": "user", "content": query_a},
        {"role": "assistant", "content": res_a["answer"]}
    ]
    query_b = "Who is the chief executive leading it and what is their background?"
    res_b = pipeline.answer_query(query=query_b, top_k=3, chat_history=chat_history)
    print(f"Follow-up Query: {query_b}")
    print(f"Answer:\n{res_b['answer']}\n")
    assert any(k in res_b["answer"].lower() for k in ["evelyn", "vance", "ceo", "mit"]), "Expected Dr. Evelyn Vance in answer"
    print(">>> PASS: Follow-up question correctly synthesized using conversation context.")

    # 8. Test Unanswerable Question (Strict Grounding)
    print("\n--- Test C: Unanswerable Question Refusal ---")
    query_c = "What is Dr. Evelyn Vance's favorite chocolate cake recipe?"
    res_c = pipeline.answer_query(query=query_c, top_k=3)
    print(f"Query: {query_c}")
    print(f"Answer:\n{res_c['answer']}\n")
    print(f"Refusal flagged: {res_c['is_refusal']}")
    assert res_c["is_refusal"] or any(m in res_c["answer"].lower() for m in ["does not contain", "cannot provide", "not mentioned"]), "Expected strict refusal"
    print(">>> PASS: Unanswerable question strictly refused without hallucination.")

    # 9. Verify PDF and DOCX extraction compatibility
    print("\n--- Test D: Multi-Format Extraction Check (DOCX & PDF) ---")
    docx_path = PROJECT_ROOT / "data" / "sample_competitor_report.docx"
    if docx_path.exists():
        with open(docx_path, "rb") as f:
            d_doc = DocumentProcessor.process_file(f.read(), filename="sample_competitor_report.docx")
            print(f"✓ DOCX Extraction: {d_doc['document_name']} -> {d_doc['total_words']} words")
            assert d_doc["total_words"] > 50

    pdf_path = PROJECT_ROOT / "data" / "sample_document.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            p_doc = DocumentProcessor.process_file(f.read(), filename="sample_document.pdf")
            print(f"✓ PDF Extraction: {p_doc['document_name']} -> {p_doc['total_words']} words, {p_doc['total_pages']} page(s)")
            assert p_doc["total_words"] > 50

    print("\n" + "=" * 70)
    print("   ALL WORKFLOW VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_workflow()
