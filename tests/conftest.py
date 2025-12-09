# tests/conftest.py
import os
import sys
from unittest.mock import MagicMock

# ==============================
# 1. Fake environment variables FIRST (before any imports)
# ==============================
os.environ["GOOGLE_API_KEY"] = "FAKE_KEY"
os.environ["LANGSMITH_API_KEY"] = "FAKE_KEY"
os.environ["GEMINI_MODEL"] = "dummy-gemini-model"
os.environ["FASTEMBED_MODEL"] = "dummy-fastembed-model"

# ==============================
# 2. Mock heavy/unavailable modules (INCLUDING LANGCHAIN)
# ==============================
mock_modules = [
    "mlflow",
    "fastembed",
    "gradio",
    "spacy",
    "torch",
    "detoxify",
    "thinc",
    "presidio_anonymizer",
    "presidio_analyzer",
    "langchain",
    "langchain.chains",
    "langchain.retrievers",
    "langchain.prompts",
    "langchain.schema",
    "langchain_core",
    "langchain_google_genai",
    "langchain_community",
    "langchain.text_splitter",
    "langchain.vectorstores",
    "langchain.embeddings",
    "langchain.chat_models",
    "prometheus_client",
]
for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# presidio_anonymizer.entities mock
sys.modules["presidio_anonymizer.entities"] = MagicMock()

# ==============================
# 3. NOW import and patch RAGConfig (after env vars are set)
# ==============================
import src.rag.config as rag_config  # noqa: E402


class DummyRAGConfig:
    """Dummy RAG configuration for testing."""

    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
    GEMINI_MODEL = os.environ["GEMINI_MODEL"]
    FASTEMBED_MODEL = os.environ["FASTEMBED_MODEL"]

    LANGSMITH_PROJECT = "test-project"
    LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    DOCUMENTS_PATH = "tests/dummy_documents"
    INDEX_PATH = "tests/dummy_index"

    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.7
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048
    RETURN_SOURCE_DOCUMENTS = True
    VERBOSE = False

    GRADIO_THEME = "soft"
    GRADIO_SHARE = False


# Patch RAGConfig class
rag_config.RAGConfig = DummyRAGConfig

# Patch module-level constants
for name in [
    "GOOGLE_API_KEY",
    "LANGSMITH_API_KEY",
    "GEMINI_MODEL",
    "FASTEMBED_MODEL",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DOCUMENTS_PATH",
    "INDEX_PATH",
    "TOP_K",
    "SIMILARITY_THRESHOLD",
    "TEMPERATURE",
    "MAX_TOKENS",
    "RETURN_SOURCE_DOCUMENTS",
    "VERBOSE",
    "GRADIO_THEME",
    "GRADIO_SHARE",
]:
    setattr(rag_config, name, getattr(DummyRAGConfig, name))

# Recreate config instance with dummy values
rag_config.config = DummyRAGConfig()

# ==============================
# 4. Mock Guardrails
# ==============================
try:
    import src.guardrails.filters.input_validator as iv_module  # noqa: E402

    iv_module.OperatorConfig = MagicMock()
    iv_module.InputValidator = MagicMock()
except Exception:
    pass
