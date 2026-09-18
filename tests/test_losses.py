from __future__ import annotations

import math
import unittest

import numpy as np

from opd.training.losses import (
    log_softmax_numpy,
    sampled_token_k2_numpy,
    sampled_token_k2_torch,
    sparse_kl_numpy,
)
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

    def test_sampled_token_k2_is_zero_when_logprobs_match(self) -> None:
        logprobs = np.asarray([-0.1, -1.2, -3.0], dtype=np.float64)
        loss, weights = sampled_token_k2_numpy(logprobs, logprobs, alpha=1.0)
        self.assertEqual(loss, 0.0)
        self.assertTrue(np.all(weights >= 1.0))
        self.assertTrue(np.all(weights < 2.0))

    def test_sure_upweights_low_student_probability_tokens(self) -> None:
        student = np.asarray([-0.1, -4.0], dtype=np.float64)
        teacher = np.asarray([-0.2, -1.0], dtype=np.float64)
        _, weights = sampled_token_k2_numpy(student, teacher, alpha=1.0)
        self.assertGreater(weights[1], weights[0])

    def test_sure_weight_branch_is_detached(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        student = torch.tensor([-0.2, -2.0], requires_grad=True)
        teacher = torch.tensor([-0.4, -1.0])
        loss, weights = sampled_token_k2_torch(
            student,
            teacher,
            alpha=1.0,
            mask=torch.ones(2),
        )
        self.assertFalse(weights.requires_grad)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_alpha_zero_matches_vanilla_value_and_gradient(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        teacher = torch.tensor([-0.3, -1.2])
        student = torch.tensor([-0.5, -1.8], requires_grad=True)
        loss, _ = sampled_token_k2_torch(
            student,
            teacher,
            alpha=0.0,
            mask=torch.ones(2),
        )
        loss.backward()
        expected = 0.5 * (teacher - student.detach()).square().mean()
        expected_gradient = (student.detach() - teacher) / 2
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertTrue(torch.allclose(student.grad, expected_gradient))

    def test_global_denominator_matches_whole_batch_loss(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        teacher = torch.tensor([-0.3, -1.2, -0.8, -2.1])
        whole_student = torch.tensor([-0.5, -1.8, -0.4, -2.5], requires_grad=True)
        whole_loss, _ = sampled_token_k2_torch(
            whole_student,
            teacher,
            alpha=1.0,
            mask=torch.ones(4),
        )
        whole_loss.backward()

        split_student = whole_student.detach().clone().requires_grad_(True)
        split_loss = torch.tensor(0.0)
        for start in (0, 2):
            part, _ = sampled_token_k2_torch(
                split_student[start : start + 2],
                teacher[start : start + 2],
                alpha=1.0,
                mask=torch.ones(2),
                denominator=torch.tensor(4.0),
            )
            split_loss = split_loss + part
        split_loss.backward()
        self.assertTrue(torch.allclose(split_loss.detach(), whole_loss.detach()))
        self.assertTrue(torch.allclose(split_student.grad, whole_student.grad))

    def test_sampled_token_k2_mask_uses_valid_token_denominator(self) -> None:
        student = np.asarray([-2.0, -2.0], dtype=np.float64)
        teacher = np.asarray([-1.0, -1.0], dtype=np.float64)
        loss, _ = sampled_token_k2_numpy(
            student,
            teacher,
            alpha=0.0,
            mask=np.asarray([1.0, 0.0]),
        )
        self.assertAlmostEqual(loss, 0.5)


if __name__ == "__main__":
    unittest.main()
