from __future__ import annotations

import unittest

from opd.config import load_config
from opd.data.prepare import _passes_filters, convert_prompt_row


class DataPrepareTest(unittest.TestCase):
    def test_deepmath_filter_accepts_only_numeric_hard_examples(self) -> None:
        config = load_config("configs/sure_k2_24h.yaml")
        self.assertTrue(_passes_filters({"difficulty": 6}, config))
        self.assertTrue(_passes_filters({"level": "7"}, config))
        self.assertFalse(_passes_filters({"difficulty": 5}, config))
        self.assertFalse(_passes_filters({"difficulty": "unknown"}, config))
        self.assertFalse(_passes_filters({}, config))

    def test_converts_openr1_default_schema(self) -> None:
        config = load_config("configs/base.yaml")
        record = convert_prompt_row(
            {
                "problem": "Find $x$ if $x+2=5$.",
                "solution": "Subtract 2 to obtain $x=3$.",
                "answer": "3",
                "problem_type": "Algebra",
                "source": "numina",
                "synthetic": False,
                "problem_is_valid": True,
                "solution_is_valid": True,
            },
            config,
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.problem, "Find $x$ if $x+2=5$.")
        self.assertEqual(record.reference_solution, "Subtract 2 to obtain $x=3$.")
        self.assertEqual(record.reference_answer, "3")
        self.assertEqual(record.subject, "Algebra")
        self.assertEqual(record.source, "numina")
        self.assertEqual(
            record.source_revision,
            "e4e141ec9dea9f8326f4d347be56105859b2bd68",
        )


if __name__ == "__main__":
    unittest.main()
