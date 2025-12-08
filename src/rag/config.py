"""
RAG Pipeline Configuration - LangChain + LangSmith + FastEmbed + Google Gemini
"""

import os
from dotenv import load_dotenv

load_dotenv()


class RAGConfig:
    """RAG Pipeline Configuration"""

    def __init__(self):
        # LangSmith
        self.LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
        self.LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-energy-assistant")
        self.LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
        if self.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
            print(f"✓ LangSmith enabled: Project '{self.LANGSMITH_PROJECT}'")
        else:
            print("⚠️  LangSmith not configured (LANGSMITH_API_KEY missing)")

        # Google AI
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

        # FastEmbed
        self.FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

        # Document processing
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
        self.DOCUMENTS_PATH = "data/documents"
        self.INDEX_PATH = "data/faiss_index"

        # Retrieval
        self.TOP_K = int(os.getenv("TOP_K", 3))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

        # LLM generation
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))

        # RAG chain
        self.RETURN_SOURCE_DOCUMENTS = True
        self.VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

        # UI
        self.GRADIO_THEME = os.getenv("GRADIO_THEME", "soft")
        self.GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

        # Validate
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables!")

        print("✓ Configuration loaded")
        print(f"  LLM: {self.GEMINI_MODEL}")
        print(f"  Embeddings: {self.FASTEMBED_MODEL} (local)")
        print(f"  LangSmith: {'✓ Enabled' if self.LANGSMITH_API_KEY else '✗ Disabled'}")


# ==============================
# Singleton instance for backward compatibility
# ==============================
config = RAGConfig()

# Export instance attributes as module-level constants for backward compatibility
GOOGLE_API_KEY = config.GOOGLE_API_KEY
LANGSMITH_API_KEY = config.LANGSMITH_API_KEY
GEMINI_MODEL = config.GEMINI_MODEL
FASTEMBED_MODEL = config.FASTEMBED_MODEL
LANGSMITH_PROJECT = config.LANGSMITH_PROJECT
LANGSMITH_ENDPOINT = config.LANGSMITH_ENDPOINT
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
DOCUMENTS_PATH = config.DOCUMENTS_PATH
INDEX_PATH = config.INDEX_PATH
TOP_K = config.TOP_K
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD
TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS
RETURN_SOURCE_DOCUMENTS = config.RETURN_SOURCE_DOCUMENTS
VERBOSE = config.VERBOSE
GRADIO_THEME = config.GRADIO_THEME
GRADIO_SHARE = config.GRADIO_SHARE
