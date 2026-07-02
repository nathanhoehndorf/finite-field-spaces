import unittest
import numpy as np

from src.operators import compare_sumset_methods


class SumsetFwhtComparisonTest(unittest.TestCase):
    def test_random_subset_matches_fwht_and_original(self):
        subset, fwht_sumset, original_sumset = compare_sumset_methods(n=6, subset_size=10, seed=7)

        self.assertEqual(subset.shape[1], 6)
        self.assertEqual(subset.shape[0], 10)
        self.assertEqual({tuple(row) for row in fwht_sumset}, {tuple(row) for row in original_sumset})


if __name__ == "__main__":
    unittest.main()
