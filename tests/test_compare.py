from __future__ import annotations

import unittest

from opd.evaluation.compare import _scores_by_sample


class EvaluationComparisonTest(unittest.TestCase):
    def test_groups_repeated_candidates_by_sample(self) -> None:
        grouped = _scores_by_sample(
            [
                {"sample_id": "a", "candidate_index": 0, "score": 0.0},
                {"sample_id": "b", "candidate_index": 0, "score": 1.0},
                {"sample_id": "a", "candidate_index": 1, "score": 1.0},
                {"sample_id": "b", "candidate_index": 1, "score": 1.0},
            ]
        )

        self.assertEqual(grouped, {"a": [0.0, 1.0], "b": [1.0, 1.0]})
        avg_at_k = sum(sum(scores) / len(scores) for scores in grouped.values()) / len(grouped)
        pass_at_k = sum(any(score > 0 for score in scores) for scores in grouped.values()) / len(
            grouped
        )
        self.assertEqual(avg_at_k, 0.75)
        self.assertEqual(pass_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
