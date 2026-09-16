from __future__ import annotations

import unittest

from pydantic import ValidationError

from opd.schemas import TeacherAnnotationRecord


class SchemaTest(unittest.TestCase):
    def test_teacher_token_shapes_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            TeacherAnnotationRecord(
                rollout_id="r1",
                sample_id="s1",
                teacher_model="teacher",
                teacher_revision="rev",
                tokenizer_revision="tok",
                tokenizer_fingerprint="fingerprint",
                input_ids=[1, 2],
                response_start=1,
                response_token_ids=[2],
                topk_token_ids=[[2]],
                topk_logprobs=[],
                tail_mass=[0.1],
                token_entropy=[0.2],
                teacher_tokens=2,
                latency_ms=1,
            )


if __name__ == "__main__":
    unittest.main()
