from __future__ import annotations

import unittest

from opd.evaluation.statistics import mcnemar_exact, paired_bootstrap


class StatisticsTest(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap([0, 1, 0, 1], [1, 1, 1, 1], iterations=500, seed=42)
        second = paired_bootstrap([0, 1, 0, 1], [1, 1, 1, 1], iterations=500, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first.mean_difference, 0.5)

    def test_mcnemar_counts_discordant_pairs(self) -> None:
        result = mcnemar_exact([True, True, False, False], [True, False, True, True])
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["candidate_only_correct"], 2)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
