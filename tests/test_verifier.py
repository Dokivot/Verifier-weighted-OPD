from __future__ import annotations

import unittest

from opd.schemas import VerificationStatus
from opd.verifier.math import MathVerifier, answers_equivalent, extract_answer


class MathVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = MathVerifier()

    def test_extracts_last_boxed_answer(self) -> None:
        self.assertEqual(extract_answer("First \\boxed{2}, finally \\boxed{3}."), "3")

    def test_extracts_nested_boxed_answer(self) -> None:
        self.assertEqual(
            extract_answer(r"After simplifying, \boxed{\frac{1}{2}}."),
            r"\frac{1}{2}",
        )

    def test_fraction_and_decimal_are_equivalent(self) -> None:
        self.assertTrue(answers_equivalent("1/2", "0.5"))

    def test_pass_fail_and_unknown(self) -> None:
        passed = self.verifier.verify(
            rollout_id="r1",
            sample_id="s1",
            response="Final answer: \\boxed{5}",
            reference_answer="5",
        )
        failed = self.verifier.verify(
            rollout_id="r2",
            sample_id="s2",
            response="Final answer: \\boxed{6}",
            reference_answer="5",
        )
        unknown = self.verifier.verify(
            rollout_id="r3",
            sample_id="s3",
            response="I tried two methods.\nNeither produced a stable result.",
            reference_answer="5",
        )
        self.assertEqual(passed.status, VerificationStatus.PASS)
        self.assertEqual(failed.status, VerificationStatus.FAIL)
        self.assertEqual(unknown.status, VerificationStatus.UNKNOWN)
        self.assertEqual(passed.verifier_version, "2")
        expected_detail_keys = {
            "comparison",
            "reason",
            "message",
            "extraction_mode",
            "extraction_confidence",
            "parser_status",
        }
        self.assertEqual(set(passed.details), expected_detail_keys)
        self.assertEqual(set(failed.details), expected_detail_keys)
        self.assertEqual(set(unknown.details), expected_detail_keys)

    def test_math_verify_tail_fallback_handles_unmarked_latex(self) -> None:
        result = self.verifier.verify(
            rollout_id="r4",
            sample_id="s4",
            response="I simplify the expression.\n$\\frac{1}{2}$",
            reference_answer="1/2",
        )
        self.assertEqual(result.status, VerificationStatus.PASS)
        self.assertEqual(result.details["extraction_mode"], "math_verify_tail_fallback")
        self.assertEqual(result.details["extraction_confidence"], "medium")

    def test_tail_fallback_handles_interval(self) -> None:
        result = self.verifier.verify(
            rollout_id="r5",
            sample_id="s5",
            response="The solution set follows.\n$(-\\infty, 2]$",
            reference_answer=r"(-\infty, 2]",
        )
        self.assertEqual(result.status, VerificationStatus.PASS)

    def test_intermediate_math_is_not_taken_from_earlier_line(self) -> None:
        result = self.verifier.verify(
            rollout_id="r6",
            sample_id="s6",
            response="An intermediate calculation is $2+3=5$.\nI cannot finish the proof.",
            reference_answer="5",
        )
        self.assertEqual(result.status, VerificationStatus.UNKNOWN)

    def test_unmarked_wrong_answer_is_fail(self) -> None:
        result = self.verifier.verify(
            rollout_id="r7",
            sample_id="s7",
            response="After checking the work, the result is\n$6$",
            reference_answer="5",
        )
        self.assertEqual(result.status, VerificationStatus.FAIL)
        self.assertEqual(result.details["extraction_mode"], "math_verify_tail_fallback")


if __name__ == "__main__":
    unittest.main()
