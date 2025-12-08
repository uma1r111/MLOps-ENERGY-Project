"""
RAG Pipeline Configuration - LangChain + LangSmith + FastEmbed + Google Gemini
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LangSmith Configuration (Token Monitoring & Tracing)
# ============================================================
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-energy-assistant")
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

# Enable LangSmith tracing
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    print(f"✓ LangSmith enabled: Project '{LANGSMITH_PROJECT}'")
else:
    print("⚠️  LangSmith not configured (LANGSMITH_API_KEY missing)")

# ============================================================
# Google AI Configuration (LLM only)
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# ============================================================
# FastEmbed Configuration (local embeddings)
# ============================================================
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ============================================================
# Document Processing
# ============================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
DOCUMENTS_PATH = "data/documents"
INDEX_PATH = "data/faiss_index"

# ============================================================
# Retrieval Settings
# ============================================================
TOP_K = int(os.getenv("TOP_K", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

# ============================================================
# LLM Generation Settings
# ============================================================
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))

# ============================================================
# RAG Chain Settings
# ============================================================
RETURN_SOURCE_DOCUMENTS = True
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# ============================================================
# UI Settings
# ============================================================
GRADIO_THEME = os.getenv("GRADIO_THEME", "soft")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# ============================================================
# Validate Required Keys
# ============================================================
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables!\n"
        "Get your API key from: https://makersuite.google.com/app/apikey"
    )

print(f"✓ Configuration loaded")
print(f"  LLM: {GEMINI_MODEL}")
print(f"  Embeddings: {FASTEMBED_MODEL} (local)")
print(f"  LangSmith: {'✓ Enabled' if LANGSMITH_API_KEY else '✗ Disabled'}")