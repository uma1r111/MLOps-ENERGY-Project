import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRAGPipeline(unittest.TestCase):

    def test_config_file_exists(self):
        config_file = Path("src/rag/config.py")
        self.assertTrue(config_file.exists(), "RAG config should exist")

    def test_ingest_file_exists(self):
        ingest_file = Path("src/rag/ingest.py")
        self.assertTrue(ingest_file.exists(), "RAG ingest should exist")

    def test_eval_dataset_exists(self):
        eval_file = Path("data/eval.jsonl")
        self.assertTrue(eval_file.exists(), "Evaluation dataset should exist")


if __name__ == "__main__":
    unittest.main()
