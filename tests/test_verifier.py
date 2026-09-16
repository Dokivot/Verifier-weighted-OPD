from __future__ import annotations

import unittest

from opd.schemas import VerificationStatus
from opd.verifier.math import MathVerifier, answers_equivalent, extract_answer


class MathVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = MathVerifier()

    def test_extracts_last_boxed_answer(self) -> None:
        self.assertEqual(extract_answer("First \\boxed{2}, finally \\boxed{3}."), "3")

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


if __name__ == "__main__":
    unittest.main()
