import unittest
from pathlib import Path


class TestPrompts(unittest.TestCase):

    def test_prompt_file_exists(self):
        prompt_file = Path("experiments/prompts.py")
        self.assertTrue(prompt_file.exists(), "prompts.py should exist")

    def test_prompt_file_not_empty(self):
        prompt_file = Path("experiments/prompts.py")
        if prompt_file.exists():
            content = prompt_file.read_text(encoding="utf-8")
            self.assertGreater(len(content), 0, "Prompt file should not be empty")

    def test_prompt_syntax(self):
        prompt_file = Path("experiments/prompts.py")
        if prompt_file.exists():
            content = prompt_file.read_text(encoding="utf-8")
            try:
                compile(content, str(prompt_file), "exec")
            except SyntaxError as e:
                self.fail(f"Syntax error in prompts.py: {e}")


if __name__ == "__main__":
    unittest.main()
