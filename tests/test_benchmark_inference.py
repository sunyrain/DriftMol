import unittest

from scripts.benchmark_inference import batches


class BenchmarkInferenceTest(unittest.TestCase):
    def test_batches_covers_total_without_overflow(self):
        self.assertEqual(list(batches(0, 4)), [])
        self.assertEqual(list(batches(3, 4)), [3])
        self.assertEqual(list(batches(10, 4)), [4, 4, 2])


if __name__ == "__main__":
    unittest.main()
