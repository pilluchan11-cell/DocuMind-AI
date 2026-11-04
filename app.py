"""
DocuMind AI – Intelligent Document Chatbot
Chat with your documents using AI.

A clean, focused, single-page Streamlit application that enables users to:
1. Upload a PDF, DOCX, or TXT document.
2. Automatically analyze and index the document locally using Sentence Transformers & FAISS.
3. Chat with the document in natural language via local Ollama (llama3.2:3b).
4. Inspect transparent source references for each answer.
"""

import hashlib
import sys
from pathlib import Path

import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.document_processor import DocumentProcessor, DocumentProcessingError
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.ollama_client import OllamaClient, OllamaClientError
from src.rag_pipeline import RAGPipeline


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS
css_path = config.ASSETS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==============================================================================
# CACHED RESOURCE INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager(model_name=config.DEFAULT_EMBEDDING_MODEL)


@st.cache_resource
def get_ollama_client() -> OllamaClient:
    return OllamaClient(
        base_url=config.OLLAMA_BASE_URL,
        default_model=config.DEFAULT_OLLAMA_MODEL
    )


def format_file_size(size_bytes: int) -> str:
    """Formats file size into human-readable B, KB, or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)

if "current_document" not in st.session_state:
    st.session_state.current_document = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Load singletons
embedding_mgr = get_embedding_manager()
ollama_client = get_ollama_client()


# ==============================================================================
# A. HEADER
# ==============================================================================
st.markdown(
    f"""
    <div class="documind-header">
        <div class="documind-title">{config.APP_ICON} {config.APP_NAME}</div>
        <div class="documind-subtitle">{config.APP_TAGLINE}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Verify local Ollama status
is_connected, msg, installed_models = ollama_client.check_connection()
if not is_connected:
    st.error(
        "⚠️ **Local Ollama service is not running.**\n\n"
        "DocuMind AI requires Ollama to generate answers locally on your machine.\n\n"
        "**To start Ollama:**\n"
        "1. Open PowerShell or a terminal.\n"
        "2. Run: `ollama serve` (or launch Ollama from your Start menu).\n"
        "3. Refresh this page."
    )
    st.stop()

model_installed, _ = ollama_client.is_model_installed(config.DEFAULT_OLLAMA_MODEL)
if not model_installed:
    st.warning(
        f"⚠️ **Model `{config.DEFAULT_OLLAMA_MODEL}` is not installed in Ollama.**\n\n"
        f"Installed models: {', '.join(installed_models) if installed_models else 'None'}\n\n"
        f"**To install the model, open a terminal and run:**\n"
        f"```powershell\nollama pull {config.DEFAULT_OLLAMA_MODEL}\n```\n"
        "Once the pull finishes, refresh this page."
    )
    st.stop()


# ==============================================================================
# B. DOCUMENT UPLOAD SECTION
# ==============================================================================
st.markdown("### 📄 Document Upload")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    help="Upload your document to read and chat with it",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # 1. Safely read file bytes and reset pointer
    try:
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        else:
            file_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {str(e)}")
        file_bytes = b""

    # Validate non-empty file
    if not file_bytes or len(file_bytes.strip()) == 0:
        st.error(f"❌ Uploaded file **{uploaded_file.name}** is empty (0 bytes). Please upload a valid document.")
    else:
        file_size_str = format_file_size(len(file_bytes))
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Display uploaded filename and size
        st.caption(f"📎 **Uploaded File:** `{uploaded_file.name}` ({file_size_str})")

        # Check if already processed in this session
        is_already_active = (
            st.session_state.current_document is not None
            and st.session_state.current_document.get("file_hash") == file_hash
        )

        if not is_already_active:
            with st.status(f"Processing '{uploaded_file.name}'...", expanded=True) as status:
                try:
                    status.write("• Reading and extracting document text...")
                    processed_doc = DocumentProcessor.process_file(file_bytes, filename=uploaded_file.name)

                    status.write("• Segmenting text into overlapping chunks...")
                    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                    chunks = chunker.chunk_document(processed_doc)

                    status.write(f"• Generating embeddings & indexing in FAISS ({len(chunks)} chunks)...")
                    embeddings = embedding_mgr.embed_chunks(chunks)

                    fresh_store = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
                    fresh_store.add_chunks(chunks, embeddings)
                    st.session_state.vector_store = fresh_store

                    # Update current document state
                    st.session_state.current_document = {
                        "filename": uploaded_file.name,
                        "file_size": file_size_str,
                        "file_hash": file_hash,
                        "file_type": processed_doc["file_type"],
                        "total_pages": processed_doc["total_pages"],
                        "total_chunks": len(chunks),
                        "total_words": processed_doc["total_words"]
                    }
                    # Reset chat history for newly uploaded document
                    st.session_state.chat_history = []

                    status.update(label="Document processed successfully! Ready to chat.", state="complete", expanded=False)
                    st.success(f"✓ **{uploaded_file.name}** processed successfully! You can now ask questions below.")

                except DocumentProcessingError as e:
                    status.update(label="Document processing failed", state="error", expanded=True)
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    status.update(label="Unexpected error", state="error", expanded=True)
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
        else:
            st.success(f"✓ Document **{uploaded_file.name}** is loaded and ready.")
else:
    # Clear state when file is removed
    if st.session_state.current_document is not None:
        st.session_state.current_document = None
        st.session_state.vector_store = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
        st.session_state.chat_history = []


# ==============================================================================
# C. DOCUMENT INFORMATION
# ==============================================================================
if st.session_state.current_document is not None:
    doc_info = st.session_state.current_document
    pages_display = f"{doc_info['total_pages']} pages" if doc_info['total_pages'] is not None else "Not applicable"

    st.markdown("#### ℹ️ Document Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Document Name", doc_info["filename"])
    with c2:
        st.metric("File Type", doc_info["file_type"])
    with c3:
        st.metric("Pages", pages_display)
    with c4:
        st.metric("Text Chunks", f"{doc_info['total_chunks']} chunks")


# ==============================================================================
# D. CHAT INTERFACE & E. SOURCE REFERENCES
# ==============================================================================
st.markdown("---")
st.markdown("### 💬 Chat with Your Document")

if st.session_state.current_document is None:
    st.info("👆 Please upload a PDF, DOCX, or TXT document above to start asking questions.")
else:
    # Toolbar with Clear Chat button
    col_status, col_clear = st.columns([4, 1])
    with col_status:
        st.caption(f"Status: **Ready to chat** | Active: `{st.session_state.current_document['filename']}`")
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear conversation history"):
            st.session_state.chat_history = []
            st.rerun()

    # Render previous conversation turns
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                sources = message.get("sources", [])
                if sources:
                    with st.expander("📄 View Source References"):
                        for idx, src in enumerate(sources, 1):
                            page_label = f"Page {src['page_number']}" if src.get("page_number") is not None else "Page N/A"
                            st.markdown(f"**Source {idx}:** `{src.get('document_name')}` | {page_label}")
                            st.markdown(f"> {src.get('text')}")

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        # Display and record user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate intelligent response
        with st.chat_message("assistant"):
            retriever = Retriever(
                embedding_manager=embedding_mgr,
                vector_store=st.session_state.vector_store
            )
            pipeline = RAGPipeline(
                retriever=retriever,
                ollama_client=ollama_client
            )

            # Stream tokens in real time
            token_stream, sources = pipeline.answer_query_stream(
                query=user_query,
                top_k=config.TOP_K_CHUNKS,
                chat_history=st.session_state.chat_history[:-1],
                max_tokens=config.OLLAMA_MAX_TOKENS
            )

            # Display streamed tokens
            full_answer = st.write_stream(token_stream)

            # Display source references below response
            if sources:
                with st.expander("📄 View Source References"):
                    for idx, src in enumerate(sources, 1):
                        page_label = f"Page {src['page_number']}" if src.get("page_number") is not None else "Page N/A"
                        st.markdown(f"**Source {idx}:** `{src.get('document_name')}` | {page_label}")
                        st.markdown(f"> {src.get('text')}")

            # Append assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })
