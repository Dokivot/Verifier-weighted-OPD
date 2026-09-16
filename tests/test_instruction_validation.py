from __future__ import annotations

import unittest

from opd.training.hf import _instruction_constraints_pass


class InstructionValidationTest(unittest.TestCase):
    def test_constraint_evaluator_is_deterministic(self) -> None:
        constraints = {
            "starts_with": "Result:",
            "required_substrings": ["prime"],
            "forbidden_substrings": ["composite"],
            "max_words": 8,
        }
        self.assertTrue(_instruction_constraints_pass("Result: seven is prime", constraints))
        self.assertFalse(
            _instruction_constraints_pass("Result: seven is composite, not prime", constraints)
        )


if __name__ == "__main__":
    unittest.main()
