import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestGuardrailsIntegration(unittest.TestCase):

    def test_guardrail_config_exists(self):
        config_file = Path("config/guardrails/guardrails_config.json")
        self.assertTrue(config_file.exists(), "Guardrails config should exist")

    def test_guardrail_engine_file_exists(self):
        engine_file = Path("src/guardrails/guardrail_engine.py")
        self.assertTrue(engine_file.exists(), "Guardrail engine should exist")

    def test_rag_integration_exists(self):
        integration_file = Path("src/guardrails/rag_integration.py")
        self.assertTrue(integration_file.exists(), "RAG integration should exist")


if __name__ == "__main__":
    unittest.main()
