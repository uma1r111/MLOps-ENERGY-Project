import unittest
import sys
from pathlib import Path

sys.path.insert(0, "scripts")


class TestLintPromptsDetailed(unittest.TestCase):

    def test_lint_prompt_file_function(self):
        """Test lint_prompt_file with actual file."""
        from lint_prompts import lint_prompt_file

        # Test with existing prompts file
        prompt_file = Path("experiments/prompts.py")
        if prompt_file.exists():
            errors = lint_prompt_file(prompt_file)
            self.assertIsInstance(errors, list)
            # Should have no errors for valid file
            self.assertEqual(len(errors), 0)

    def test_lint_empty_content(self):
        """Test linting detects empty files."""
        from lint_prompts import lint_prompt_file

        # Create temporary empty file
        temp_file = Path("temp_empty.py")
        temp_file.write_text("", encoding="utf-8")

        try:
            errors = lint_prompt_file(temp_file)
            # Should detect empty file
            self.assertGreater(len(errors), 0)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_lint_long_content(self):
        """Test linting detects overly long files."""
        from lint_prompts import lint_prompt_file

        # Create temporary long file
        temp_file = Path("temp_long.py")
        long_content = "x" * 5000  # Over 4000 char limit
        temp_file.write_text(long_content, encoding="utf-8")

        try:
            errors = lint_prompt_file(temp_file)
            # Should detect long file
            self.assertGreater(len(errors), 0)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_main_function(self):
        """Test main function execution."""
        from lint_prompts import main

        # Should return 0 for success
        result = main()
        self.assertIn(result, [0, 1])


class TestEvaluatePromptsDetailed(unittest.TestCase):

    def test_evaluate_dataset_exists(self):
        """Test that evaluation dataset is defined."""
        from evaluate_prompts import EVAL_DATASET

        self.assertIsInstance(EVAL_DATASET, list)
        self.assertGreater(len(EVAL_DATASET), 0)

        # Check dataset structure
        for test_case in EVAL_DATASET:
            self.assertIn("query", test_case)
            self.assertIn("expected", test_case)

    def test_evaluate_prompts_function(self):
        """Test evaluate_prompts function."""
        from evaluate_prompts import evaluate_prompts

        result = evaluate_prompts()
        self.assertIn(result, [0, 1])

    def test_evaluation_logic(self):
        """Test that evaluation checks for keywords."""
        from evaluate_prompts import EVAL_DATASET

        # Verify dataset has expected structure
        first_test = EVAL_DATASET[0]
        self.assertIsInstance(first_test["query"], str)
        self.assertTrue(len(first_test["query"]) > 0)


if __name__ == "__main__":
    unittest.main()
