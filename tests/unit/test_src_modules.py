import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSrcModules(unittest.TestCase):

    def test_src_folder_exists(self):
        src = Path("src")
        self.assertTrue(src.exists())

    def test_rag_folder_exists(self):
        rag = Path("src/rag")
        self.assertTrue(rag.exists())

    def test_guardrails_folder_exists(self):
        guardrails = Path("src/guardrails")
        self.assertTrue(guardrails.exists())

    def test_monitoring_folder_exists(self):
        monitoring = Path("src/monitoring")
        self.assertTrue(monitoring.exists())

    def test_config_file_exists(self):
        config = Path("src/rag/config.py")
        self.assertTrue(config.exists())

    def test_ingest_file_exists(self):
        ingest = Path("src/rag/ingest.py")
        self.assertTrue(ingest.exists())


if __name__ == "__main__":
    unittest.main()
