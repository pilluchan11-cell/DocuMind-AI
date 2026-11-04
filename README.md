# DocuMind AI – Intelligent Document Chatbot
### Major Practical Assignment: Prompt Engineering

> **DocuMind AI** – Chat with your documents using AI.

A clean, reliable, single-page document-based AI chatbot built with Streamlit, Sentence Transformers, FAISS, and local Ollama (`llama3.2:3b`).

---

## 1. Project Overview

DocuMind AI enables users to:
1. **Upload their own document** in PDF (`.pdf`), Word (`.docx`), or Text (`.txt`) format.
2. **Automatically analyze and index** the document locally (extraction, sentence-aware chunking, dense vector embeddings, FAISS CPU index).
3. **Ask questions in natural language**, including paraphrased queries with different wording from the source document.
4. **Receive intelligent, grounded answers** in fluent English without robotic output or hallucinated facts.
5. **Inspect transparent source references** with document name, page number (where available), and exact retrieved text passages.

### Strict Privacy & Local Execution Rules
- **100% Local Inference**: Uses Ollama with `llama3.2:3b`.
- **Zero Paid APIs**: No OpenAI, No Gemini, No Claude, and no paid API keys.
- **Genuine RAG Pipeline**: No simulated similarity search, no mock retrieval, and no hardcoded chatbot responses.
- **Lightweight Architecture**: Pure Python modular design without unnecessary orchestration overhead or complex dashboards.

---

## 2. Technology Stack

| Component | Technology | Role / Specification |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core application logic and pipeline |
| **Frontend Framework** | Streamlit | Single-page conversational UI |
| **PDF Extraction** | PyMuPDF (`pymupdf` / `fitz`) | Stream-based page-by-page extraction with page numbers |
| **DOCX Extraction** | `python-docx` | Paragraphs and tables (without inventing fake page numbers) |
| **TXT Extraction** | Python standard library | Multi-encoding decoding (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`) |
| **Embedding Model** | Sentence Transformers (`all-MiniLM-L6-v2`) | 384-dimensional dense vectors with L2 normalization |
| **Vector Database** | FAISS CPU (`IndexFlatIP`) | Cosine similarity via inner product of normalized vectors |
| **Local LLM** | Ollama (`llama3.2:3b`) | Local inference via `http://localhost:11434/api/chat` |

---

## 3. Architecture & RAG Workflow

```
[User Uploads Document: PDF / DOCX / TXT]
                     ↓
[Safe Stream Byte Processing (DocumentProcessor)]
                     ↓
[Sentence-Aware Chunking (TextChunker: ~600 tokens, ~100 overlap)]
                     ↓
[Sentence Transformers: all-MiniLM-L6-v2 (L2-Normalized 384-d vectors)]
                     ↓
[FAISS CPU Vector Store (IndexFlatIP Cosine Similarity Index)]
                     ↓
[User Asks Question in Natural Language]
                     ↓
[Query Embedding Generation (Same all-MiniLM-L6-v2 model)]
                     ↓
[FAISS Semantic Search: Top 4 Relevant Chunks]
                     ↓
[Prompt Engineering: System Prompt with 6 Anti-Hallucination Rules]
                     ↓
[Local Ollama /api/chat: llama3.2:3b Inference]
                     ↓
[Streamed Answer + Expandable Original Source References]
```

---

## 4. Prompt Engineering & Hallucination Prevention

DocuMind AI incorporates a carefully crafted system prompt passed to Ollama `/api/chat`:

```text
You are DocuMind AI, an intelligent document assistant.

Your responsibility is to answer questions based on the information retrieved from the user's uploaded document.

You must follow these rules:

1. Understand the user's question, even if it is phrased differently from the original document.
2. Use the supplied document context as the factual basis of your answer.
3. Do not invent facts, figures, dates, names or statistics.
4. Do not use outside knowledge to fill missing information.
5. If the answer cannot be found in the supplied context, clearly state that the document does not contain sufficient information to answer the question, and politely decline to speculate.
6. Provide clear, direct, and well-structured answers.
```

---

## 5. User Interface (Single-Page Structure)

The interface is intentionally simple, elegant, and focused:
- **A. Header**: Logo icon (`🧠`), application title (`DocuMind AI`), and subtitle (`Chat with your documents using AI.`).
- **B. Document Upload Section**: File uploader supporting `.pdf`, `.docx`, and `.txt`, displaying the uploaded filename, formatted file size (e.g. `42.5 KB`), processing status, and success confirmation.
- **C. Document Information**: Summary displaying Document Name, File Type, Pages (where available, e.g. for PDF, or `Not applicable` for DOCX/TXT), Text Chunks, and Status (`Ready to chat`).
- **D. Chat Interface**: Conversational message flow, real-time token streaming, question text input, and a `🗑️ Clear Chat` button.
- **E. Source References**: Below each AI response, an expandable `📄 View Source References` panel displays the document name, page number (where available), and exact retrieved text excerpts.

---

## 6. How to Run

### Step 1: Ensure Ollama is Running with `llama3.2:3b`
```powershell
# In PowerShell or terminal:
ollama serve

# Verify or pull the required model:
ollama pull llama3.2:3b
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Launch DocuMind AI
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Running the Automated Test Suite

A comprehensive 10-test suite validates all requirements:
```powershell
python -m unittest tests/test_documind.py
```

### Test Coverage Summary:
- `test_01_ollama_service_and_model`: Checks Ollama connection and `llama3.2:3b` presence.
- `test_02_pdf_extraction_and_page_preservation`: Streamlit `UploadedFile` byte safety, PyMuPDF extraction, and 1-indexed page numbers.
- `test_03_docx_extraction_no_invented_pages`: python-docx extraction of paragraphs and tables (`page_number = None`).
- `test_04_txt_extraction_multi_encoding`: Plain text extraction across UTF-8, UTF-8 BOM, and Latin-1.
- `test_05_error_handling_invalid_uploads`: Safe rejection of 0-byte, corrupted, and unsupported files.
- `test_06_text_chunking_with_overlap_and_metadata`: Overlapping chunk generation with chunk IDs and document attribution.
- `test_07_dense_embeddings_and_faiss_retrieval`: 384-dimensional dense vectors and cosine similarity retrieval.
- `test_08_paraphrased_rag_query_understanding`: Retrieval and answering when user query uses different wording from source text.
- `test_09_multi_turn_conversational_follow_up`: Context-aware follow-up question answering.
- `test_10_strict_grounding_refusal`: Hallucination prevention and refusal of out-of-scope questions.

---

## 8. Viva Practical Questions & Answers

### Q1: Why use RAG instead of feeding the whole document into an LLM?
**Answer:** While LLMs have context limits, large context windows are slow, computationally expensive on local CPUs, and prone to the "lost in the middle" problem. RAG segments documents into dense embeddings and retrieves only the top-K semantically relevant chunks into the prompt, resulting in fast, precise, and verifiable answers.

### Q2: Why normalize embeddings for FAISS `IndexFlatIP`?
**Answer:** The cosine similarity between two vectors $A$ and $B$ is:
$$\cos(\theta) = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$
By L2-normalizing all vectors ($\|A\|_2 = 1$), cosine similarity simplifies directly to the inner product ($A \cdot B$). FAISS's `IndexFlatIP` computes the inner product rapidly on CPU.

### Q3: Why does DOCX extraction not invent page numbers?
**Answer:** Unlike PDF files which have rigid physical page structures, DOCX files store content as a continuous XML document tree of paragraphs and tables. Pagination in Word is dynamic and depends on font rendering, margins, and printer drivers. Inventing arbitrary page numbers is misleading; instead, DocuMind AI records `page_number = None` for DOCX and displays "Page N/A".
