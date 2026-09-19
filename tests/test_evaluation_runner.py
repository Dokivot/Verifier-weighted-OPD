from __future__ import annotations

import unittest

from opd.evaluation.runner import _evaluation_prompt


class EvaluationRunnerTest(unittest.TestCase):
    def test_prompt_template_preserves_latex_braces(self) -> None:
        evaluation = {
            "prompt_template": (
                "{problem}\nPlease put the answer in \\boxed{} and preserve \\frac{1}{2}."
            ),
            "generation": {},
        }
        self.assertEqual(
            _evaluation_prompt("Compute 1 + 1.", evaluation),
            "Compute 1 + 1.\nPlease put the answer in \\boxed{} and preserve \\frac{1}{2}.",
        )


if __name__ == "__main__":
    unittest.main()
