from __future__ import annotations

import math
import unittest

import numpy as np

from opd.training.losses import log_softmax_numpy, sparse_kl_numpy
from opd.training.weighting import confidence_weights


class LossTest(unittest.TestCase):
    def test_sparse_kl_matches_aggregated_full_distribution(self) -> None:
        logits = np.asarray([[1.2, -0.5, 0.3, 0.9]], dtype=np.float64)
        teacher = np.asarray([0.5, 0.2, 0.2, 0.1], dtype=np.float64)
        top_ids = np.asarray([[0, 1]])
        top_logprobs = np.log(teacher[top_ids])
        tail = np.asarray([teacher[2:].sum()])
        actual = sparse_kl_numpy(logits, top_ids, top_logprobs, tail)

        student = np.exp(log_softmax_numpy(logits))[0]
        expected = 0.0
        for teacher_probability, student_probability in zip(
            [teacher[0], teacher[1], teacher[2:].sum()],
            [student[0], student[1], student[2:].sum()],
            strict=True,
        ):
            expected += teacher_probability * math.log(teacher_probability / student_probability)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_confidence_weights_are_normalized(self) -> None:
        weights = confidence_weights([0.1, 1.0, 2.0], vocab_size=256)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, places=12)
        self.assertTrue(all(weight > 0 for weight in weights))


if __name__ == "__main__":
    unittest.main()
