import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRAGComponents(unittest.TestCase):

    def test_import_rag_config(self):
        try:
            from src.rag.config import RAGConfig

            self.assertIsNotNone(RAGConfig)
        except ImportError as e:
            self.skipTest(f"RAG config not available: {e}")

    def test_import_rag_chain(self):
        try:
            from src.rag.rag_chain import RAGChain

            self.assertIsNotNone(RAGChain)
        except ImportError as e:
            self.skipTest(f"RAG chain not available: {e}")

    def test_config_file_structure(self):
        """Test config file has required attributes."""
        from src.rag import config

        required_attrs = ["GOOGLE_API_KEY", "GEMINI_MODEL", "INDEX_PATH", "TOP_K"]
        for attr in required_attrs:
            self.assertTrue(hasattr(config, attr), f"Config missing {attr}")


if __name__ == "__main__":
    unittest.main()
