from __future__ import annotations

import unittest

from opd.training.masking import build_response_labels, response_logit_bounds


class MaskingTest(unittest.TestCase):
    def test_prompt_tokens_are_ignored_and_eos_is_supervised(self) -> None:
        input_ids, labels = build_response_labels([10, 11], [20, 21], eos_token_id=2, max_length=8)
        self.assertEqual(input_ids, [10, 11, 20, 21, 2])
        self.assertEqual(labels, [-100, -100, 20, 21, 2])

    def test_truncation_preserves_mask_alignment(self) -> None:
        input_ids, labels = build_response_labels(
            [10, 11], [20, 21, 22], eos_token_id=2, max_length=4
        )
        self.assertEqual(input_ids, [10, 11, 20, 21])
        self.assertEqual(labels, [-100, -100, 20, 21])

    def test_response_logits_are_shifted_one_position(self) -> None:
        self.assertEqual(
            response_logit_bounds(response_start=3, response_tokens=2, input_length=5),
            (2, 4),
        )
        with self.assertRaises(ValueError):
            response_logit_bounds(response_start=3, response_tokens=3, input_length=5)


if __name__ == "__main__":
    unittest.main()
