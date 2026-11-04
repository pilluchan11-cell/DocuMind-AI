"""
Configuration Module for DocuMind AI.
Centralizes paths, model parameters, chunking configurations, and thresholds.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# Application Branding
APP_NAME = "DocuMind AI"
APP_TAGLINE = "Chat with your documents using AI."
APP_ICON = "🧠"

# Supported Formats
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Embedding Model (Sentence Transformers)
# Runs locally on CPU with zero paid APIs
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Document Chunking Settings
CHUNK_SIZE = 600      # ~150 words
CHUNK_OVERLAP = 100   # Overlap between adjacent chunks

# FAISS Retrieval Settings
TOP_K_CHUNKS = 4      # Top 4-5 relevant chunks per question

# Ollama LLM Settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 120   # seconds
OLLAMA_MAX_TOKENS = 400
TEMPERATURE = 0.1
