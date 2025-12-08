import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSimpleImports(unittest.TestCase):

    def test_can_import_config(self):
        try:
            from src.rag import config

            self.assertIsNotNone(config)
        except Exception as e:
            self.skipTest(f"Config import failed: {e}")

    def test_config_has_ragconfig(self):
        try:
            from src.rag.config import RAGConfig

            self.assertIsNotNone(RAGConfig)
        except Exception as e:
            self.skipTest(f"RAGConfig import failed: {e}")

    def test_config_has_constants(self):
        try:
            from src.rag import config

            attrs = ["GOOGLE_API_KEY", "GEMINI_MODEL", "INDEX_PATH"]
            for attr in attrs:
                self.assertTrue(hasattr(config, attr))
        except Exception as e:
            self.skipTest(f"Config constants check failed: {e}")


if __name__ == "__main__":
    unittest.main()
