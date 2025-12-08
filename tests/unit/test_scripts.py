import unittest
from pathlib import Path


class TestScripts(unittest.TestCase):
    def test_lint_script_exists(self):
        script = Path("scripts/lint_prompts.py")
        self.assertTrue(script.exists())

    def test_eval_script_exists(self):
        script = Path("scripts/evaluate_prompts.py")
        self.assertTrue(script.exists())


if __name__ == "__main__":
    unittest.main()
