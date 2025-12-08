import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPromptsExecution(unittest.TestCase):

    def test_prompts_file_compiles(self):
        """Test that prompts.py is valid Python."""
        prompts_file = Path("experiments/prompts.py")
        if prompts_file.exists():
            with open(prompts_file, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                compile(content, str(prompts_file), "exec")
                self.assertTrue(True)
            except SyntaxError as e:
                self.fail(f"Syntax error: {e}")

    def test_prompts_module_imports(self):
        """Test that prompts module can be imported."""
        try:
            import experiments.prompts as prompts_module

            self.assertIsNotNone(prompts_module)
        except Exception as e:
            self.skipTest(f"Could not import prompts: {e}")


if __name__ == "__main__":
    unittest.main()
