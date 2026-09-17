from __future__ import annotations

import unittest

from opd.training.hf import _needs_final_validation


class FinalValidationTest(unittest.TestCase):
    def test_runs_when_epoch_ends_before_first_interval(self) -> None:
        self.assertTrue(
            _needs_final_validation(has_validation=True, global_step=37, eval_steps=100)
        )

    def test_skips_duplicate_scheduled_validation(self) -> None:
        self.assertFalse(
            _needs_final_validation(has_validation=True, global_step=100, eval_steps=100)
        )

    def test_skips_when_no_step_or_validation_set(self) -> None:
        self.assertFalse(
            _needs_final_validation(has_validation=True, global_step=0, eval_steps=100)
        )
        self.assertFalse(
            _needs_final_validation(has_validation=False, global_step=37, eval_steps=100)
        )


if __name__ == "__main__":
    unittest.main()
