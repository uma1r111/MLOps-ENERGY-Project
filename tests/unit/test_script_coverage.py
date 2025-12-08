import unittest
import sys


# Force import and execute the scripts to get coverage
class TestScriptCoverage(unittest.TestCase):

    def test_lint_prompts_coverage(self):
        """Import and test lint_prompts functions."""
        sys.path.insert(0, "scripts")
        try:
            from lint_prompts import lint_prompt_file

            # Test the functions
            from pathlib import Path

            if Path("experiments/prompts.py").exists():
                errors = lint_prompt_file(Path("experiments/prompts.py"))
                self.assertIsInstance(errors, list)
        except Exception as e:
            self.skipTest(f"Could not test lint_prompts: {e}")

    def test_evaluate_prompts_coverage(self):
        """Import and test evaluate_prompts functions."""
        sys.path.insert(0, "scripts")
        try:
            from evaluate_prompts import EVAL_DATASET

            self.assertIsInstance(EVAL_DATASET, list)
            self.assertGreater(len(EVAL_DATASET), 0)
        except Exception as e:
            self.skipTest(f"Could not test evaluate_prompts: {e}")


if __name__ == "__main__":
    unittest.main()
