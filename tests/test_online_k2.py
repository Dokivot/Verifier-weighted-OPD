from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from opd.hashing import file_sha256, stable_hash
from opd.schemas import PromptRecord
from opd.training.online_k2 import (
    OnlineTrajectory,
    _checkpoint_storage_plan,
    _classify_generated_tokens,
    _format_problem,
    _sampled_logprobs,
    _student_checkpoint_override,
    _stop_token_ids,
    _trim_generated_tokens,
    _validate_step_artifacts,
    normalized_online_training_config,
    online_run_id,
)


class OnlineK2Test(unittest.TestCase):
    def test_prompt_template_preserves_latex_braces(self) -> None:
        template = "{problem}\nPlease put the answer in \\boxed{} and preserve \\frac{1}{2}."
        self.assertEqual(
            _format_problem("Compute 1 + 1.", template),
            "Compute 1 + 1.\nPlease put the answer in \\boxed{} and preserve \\frac{1}{2}.",
        )

    def test_sampled_logprobs_use_shifted_response_positions(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(5))

            def forward(self, input_ids: object, **_: object) -> object:
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Backbone()
                self.lm_head = torch.nn.Linear(5, 5, bias=False)
                self.lm_head.weight.data.copy_(torch.eye(5))

        prompt = PromptRecord(
            sample_id="sample",
            problem="problem",
            reference_answer="answer",
            source="fixture",
            source_dataset="fixture",
            source_revision="v1",
            split="train",
        )
        trajectory = OnlineTrajectory(
            prompt=prompt,
            rendered_prompt="problem",
            prompt_token_ids=[0, 1],
            response_token_ids=[2, 3],
            response="response",
            seed=42,
            truncated=False,
            finish_reason="stop",
            terminal_token_id=3,
        )
        model = Model()
        actual = _sampled_logprobs(
            model,
            [trajectory],
            micro_batch_size=1,
            torch=torch,
            device=torch.device("cpu"),
            pad_token_id=4,
            require_grad=True,
            compute_dtype=torch.float32,
        )[0]
        logits = torch.eye(5)[torch.tensor([1, 2])]
        expected = torch.log_softmax(logits, dim=-1)[torch.arange(2), torch.tensor([2, 3])]
        self.assertTrue(torch.allclose(actual, expected))

    def test_eos_is_supervised_when_pad_and_eos_share_an_id(self) -> None:
        self.assertEqual(
            _trim_generated_tokens([10, 2, 2], eos_token_id=2, pad_token_id=2),
            [10, 2],
        )

    def test_qwen_stop_tokens_include_chat_and_default_eos(self) -> None:
        tokenizer = SimpleNamespace(
            eos_token_id=151643,
            unk_token_id=0,
            convert_tokens_to_ids=lambda token: {
                "<|im_end|>": 151645,
                "<|endoftext|>": 151643,
            }[token],
        )
        self.assertEqual(_stop_token_ids(tokenizer), (151643, 151645))

    def test_chat_terminator_is_kept_as_a_supervised_response_token(self) -> None:
        response_ids, finish_reason, terminal_token_id = _classify_generated_tokens(
            [17, 151645, 0, 0],
            stop_token_ids=(151643, 151645),
            pad_token_id=0,
        )
        self.assertEqual(response_ids, [17, 151645])
        self.assertEqual(finish_reason, "stop")
        self.assertEqual(terminal_token_id, 151645)

    def test_checkpoint_storage_plan_accounts_for_atomic_rolling_replacement(self) -> None:
        gib = 1024**3
        plan = _checkpoint_storage_plan(
            parameter_count=1_700_000_000,
            max_steps=55,
            milestone_steps={34, 55},
            checkpointing={
                "minimum_free_disk_gib": 80,
                "reserve_artifact_gib": 28,
                "safety_factor": 1.1,
            },
            free_bytes=100 * gib,
        )
        self.assertEqual(plan["atomic_rolling_copies"], 2)
        self.assertEqual(plan["retained_milestone_model_checkpoints"], 1)
        self.assertTrue(plan["passed"])

    def test_resume_storage_plan_does_not_charge_existing_rolling_twice(self) -> None:
        gib = 1024**3
        common = {
            "parameter_count": 1_700_000_000,
            "max_steps": 14,
            "milestone_steps": {7},
            "checkpointing": {
                "minimum_free_disk_gib": 80,
                "reserve_artifact_gib": 28,
                "safety_factor": 1.1,
            },
            "free_bytes": 84 * gib,
        }
        fresh = _checkpoint_storage_plan(**common)
        resumed = _checkpoint_storage_plan(
            **common,
            preallocated_rolling_copies=1,
        )

        self.assertFalse(fresh["passed"])
        self.assertTrue(resumed["passed"])
        self.assertEqual(resumed["atomic_rolling_copies"], 2)
        self.assertEqual(resumed["preallocated_rolling_copies"], 1)
        self.assertEqual(resumed["rolling_copies_requiring_free_space"], 1)
        self.assertEqual(resumed["required_free_disk_gib"], 80)

    def test_resume_path_and_invocation_limit_do_not_change_run_id(self) -> None:
        training = {
            "method": "sure_k2",
            "max_steps": 55,
            "resume_from_checkpoint": None,
            "invocation_step_limit": 1,
        }
        resumed = {
            **training,
            "resume_from_checkpoint": "checkpoint/rolling",
            "invocation_step_limit": 2,
        }
        common = {
            "student_config": {"name": "student", "revision": "s1"},
            "teacher_config": {"name": "teacher", "revision": "t1"},
            "input_checksum": "data",
            "seed": 42,
        }
        self.assertEqual(
            online_run_id(training=training, **common),
            online_run_id(training=resumed, **common),
        )
        self.assertNotIn("resume_from_checkpoint", normalized_online_training_config(resumed))

    def test_resume_checkpoint_takes_precedence_over_initial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rolling = Path(directory) / "rolling"
            self.assertEqual(
                _student_checkpoint_override(
                    resume_dir=rolling,
                    initial_checkpoint=Path(directory) / "missing-initial",
                ),
                rolling / "model",
            )

    def test_initial_checkpoint_is_used_for_first_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / "initial"
            initial.mkdir()
            self.assertEqual(
                _student_checkpoint_override(
                    resume_dir=None,
                    initial_checkpoint=initial,
                ),
                initial,
            )

    def test_resume_rejects_orphaned_step_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            step_dir = Path(directory)
            (step_dir / "step_000001.parquet").write_bytes(b"step-one")
            with self.assertRaisesRegex(RuntimeError, "stale or orphaned"):
                _validate_step_artifacts(
                    step_dir,
                    global_step=0,
                    history=[],
                    policy_hash="initial",
                )

    def test_resume_accepts_valid_policy_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            step_dir = Path(directory)
            path = step_dir / "step_000001.parquet"
            path.write_bytes(b"step-one")
            parent = "policy-zero"
            next_hash = stable_hash(
                {"parent": parent, "step": 1, "step_artifact": file_sha256(path)},
                length=24,
            )
            _validate_step_artifacts(
                step_dir,
                global_step=1,
                history=[
                    {
                        "step": 1,
                        "policy_hash": parent,
                        "next_policy_hash": next_hash,
                    }
                ],
                policy_hash=next_hash,
            )


if __name__ == "__main__":
    unittest.main()
