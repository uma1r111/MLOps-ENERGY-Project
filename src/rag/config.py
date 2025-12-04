"""
RAG Pipeline Configuration - FastEmbed + Google Gemini
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Google AI Configuration (LLM only)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# FastEmbed Configuration (local embeddings)
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
DOCUMENTS_PATH = "data/documents"
INDEX_PATH = "data/faiss_index"

# Retrieval Settings
TOP_K = int(os.getenv("TOP_K", 3))
SIMILARITY_THRESHOLD = 0.7

# LLM Generation Settings
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))

# Validate API Key
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables!\n"
        "Get your API key from: https://makersuite.google.com/app/apikey"
    )

print(f"✓ Configuration loaded")
print(f"  LLM: {GEMINI_MODEL}")
print(f"  Embeddings: {FASTEMBED_MODEL} (local, no API key needed)")