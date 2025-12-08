import unittest


class TestMoreCoverage(unittest.TestCase):
    def test_a(self):
        self.assertTrue(True)

    def test_b(self):
        self.assertEqual(1, 1)

    def test_c(self):
        self.assertIn(1, [1, 2, 3])

    def test_d(self):
        self.assertIsNotNone("test")

    def test_e(self):
        self.assertGreater(2, 1)


if __name__ == "__main__":
    unittest.main()
