import unittest
import subprocess
import sys


class TestScriptExecution(unittest.TestCase):

    def test_lint_prompts_runs(self):
        """Test that lint_prompts.py can be executed."""
        result = subprocess.run(
            [sys.executable, "scripts/lint_prompts.py"], capture_output=True, text=True
        )
        # Should exit with 0 or 1
        self.assertIn(result.returncode, [0, 1])

    def test_evaluate_prompts_runs(self):
        """Test that evaluate_prompts.py can be executed."""
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_prompts.py"],
            capture_output=True,
            text=True,
        )
        # Should exit with 0 or 1
        self.assertIn(result.returncode, [0, 1])


if __name__ == "__main__":
    unittest.main()
