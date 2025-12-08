import unittest
from pathlib import Path


class TestExperimentsModule(unittest.TestCase):

    def test_prompts_file_exists(self):
        prompts = Path("experiments/prompts.py")
        self.assertTrue(prompts.exists())

    def test_prompts_not_empty(self):
        prompts = Path("experiments/prompts.py")
        if prompts.exists():
            content = prompts.read_text(encoding="utf-8")
            self.assertGreater(len(content), 0)

    def test_experiments_folder_exists(self):
        experiments = Path("experiments")
        self.assertTrue(experiments.exists())


if __name__ == "__main__":
    unittest.main()
