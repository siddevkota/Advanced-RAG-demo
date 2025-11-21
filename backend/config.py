"""Configuration settings for the backend."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Validate API key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Use OpenAI for embeddings
USE_LOCAL_EMBEDDINGS = False
EMBEDDING_MODEL = "text-embedding-3-small"
PROVIDER = "openai"

# LLM Model (always OpenAI)
LLM_MODEL = "gpt-4.1"

# Legacy flags for compatibility
USE_OPENAI = True
USE_GEMINI = False

RERANKER_MODEL = "BAAI/bge-reranker-base"

# RAG Pipeline Parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
STAGE1_K = 30
TOP_K_RERANKED = 5

# Data Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FEEDBACK_FILE = DATA_DIR / "chatbot_feedback.json"
PDF_DIR = DATA_DIR / "pdfs"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True

# Startup Configuration
LOAD_SQUAD_DATASET = False  # Don't load SQuAD dataset, only use PDFs
VECTORSTORE_PATH = DATA_DIR / "vectorstore"  # Path to save/load vectorstore

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
VECTORSTORE_PATH.mkdir(exist_ok=True)
