from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.config import config_hash, load_config
from opd.exceptions import ConfigurationError


class ConfigTest(unittest.TestCase):
    def test_sure_k2_24h_is_isolated_from_legacy_training_config(self) -> None:
        config = load_config("configs/sure_k2_24h.yaml")
        training = config["training"]

        self.assertEqual(config["project"]["name"], "sure-k2-24h")
        self.assertEqual(training["backend"], "online_k2")
        self.assertEqual(training["method"], "sure_k2")
        for key in (
            "qlora",
            "lora",
            "early_stopping",
            "batch_size",
            "gradient_accumulation_steps",
        ):
            self.assertNotIn(key, training)

    def test_sure_k2_24h_config_matches_registered_budget(self) -> None:
        config = load_config("configs/sure_k2_24h.yaml")
        self.assertEqual(config["models"]["student"]["name"], "Qwen/Qwen3-1.7B-Base")
        self.assertEqual(config["models"]["teacher"]["name"], "Qwen/Qwen3-8B")
        self.assertEqual(config["data"]["filters"]["minimum_difficulty"], 6)
        self.assertEqual(config["training"]["method"], "sure_k2")
        self.assertEqual(config["training"]["global_prompt_batch_size"], 512)
        self.assertEqual(config["training"]["max_steps"], 55)
        self.assertEqual(config["training"]["max_response_tokens"], 8192)
        self.assertEqual(
            config["training"]["input_path"],
            "artifacts/sure_k2_24h/data/contamination/train_clean.parquet",
        )
        self.assertEqual(config["data"]["counts"]["train"], 30720)
        self.assertEqual(config["checkpointing"]["rolling_retention"], 1)
        self.assertTrue(config["training"]["require_decontaminated_input"])
        self.assertEqual(config["evaluation"]["suites"]["math500"]["num_samples"], 1)
        self.assertEqual(config["evaluation"]["suites"]["amc23"]["num_samples"], 4)
        self.assertFalse(config["training"]["enable_thinking"])
        self.assertEqual(config["training"]["thinking_marker"], "/no_think")
        self.assertEqual(config["evaluation"]["generation"]["thinking_marker"], "/no_think")
        self.assertEqual(config["benchmark"]["thinking_marker"], "/no_think")

    def test_warmstart_configs_preserve_base_identity_and_use_pilot_weights(self) -> None:
        formal = load_config("configs/sure_k2_warmstart_24h.yaml")
        pilot = load_config("configs/sure_k2_warmstart_pilot.yaml")
        smoke = load_config("configs/sure_k2_warmstart_smoke.yaml")

        self.assertEqual(
            formal["models"]["student"]["name"],
            "Qwen/Qwen3-1.7B-Base",
        )
        self.assertEqual(
            formal["training"]["initial_student_checkpoint"],
            "artifacts/sure_k2_24h/pilot/checkpoint/final",
        )
        self.assertEqual(formal["training"]["global_prompt_batch_size"], 128)
        self.assertEqual(formal["training"]["max_steps"], 14)
        self.assertEqual(formal["training"]["max_response_tokens"], 8192)
        self.assertEqual(formal["training"]["state_save_steps"], 2)
        self.assertEqual(pilot["training"]["max_steps"], 2)
        self.assertEqual(pilot["pilot"]["max_steady_step_seconds"], 4500)
        self.assertEqual(
            smoke["training"]["output_dir"],
            "artifacts/sure_k2_warmstart_24h/smoke/checkpoint",
        )

    def test_warmstart_fast_candidate_only_changes_micro_batches(self) -> None:
        formal = load_config("configs/sure_k2_warmstart_24h.yaml")
        fast = load_config("configs/sure_k2_warmstart_fast_24h.yaml")

        self.assertEqual(fast["training"]["global_prompt_batch_size"], 128)
        self.assertEqual(fast["training"]["max_steps"], 14)
        self.assertEqual(fast["training"]["max_response_tokens"], 8192)
        self.assertEqual(fast["training"]["rollout_micro_batch_size"], 4)
        self.assertEqual(fast["training"]["teacher_micro_batch_size"], 2)
        self.assertEqual(fast["training"]["student_micro_batch_size"], 2)
        self.assertEqual(fast["training"]["invocation_step_limit"], 1)
        self.assertEqual(
            fast["training"]["initial_student_checkpoint"],
            formal["training"]["initial_student_checkpoint"],
        )
        self.assertEqual(
            fast["training"]["input_path"],
            "artifacts/sure_k2_24h/data/contamination/train_clean.parquet",
        )
        self.assertEqual(fast["paths"]["data_dir"], "artifacts/sure_k2_24h/data")
        self.assertEqual(
            fast["evaluation"]["suites"]["math500"]["input_path"],
            "artifacts/sure_k2_24h/data/eval/math500.parquet",
        )
        self.assertEqual(
            fast["evaluation"]["suites"]["amc23"]["input_path"],
            "artifacts/sure_k2_24h/data/eval/amc23.parquet",
        )
        self.assertNotEqual(
            fast["training"]["output_dir"],
            formal["training"]["output_dir"],
        )

    def test_online_resume_arguments_do_not_change_canonical_config_hash(self) -> None:
        config = load_config("configs/sure_k2_pilot.yaml")
        resumed = {
            **config,
            "training": {
                **config["training"],
                "resume_from_checkpoint": "artifacts/sure_k2_24h/pilot/checkpoint/rolling",
                "invocation_step_limit": 1,
            },
        }
        self.assertEqual(config_hash(config), config_hash(resumed))

    def test_online_thinking_protocol_must_match(self) -> None:
        config = load_config("configs/sure_k2_24h.yaml")
        config["benchmark"]["enable_thinking"] = True
        with self.assertRaisesRegex(ConfigurationError, "same enable_thinking"):
            from opd.config import _validate_thinking_protocol

            _validate_thinking_protocol(config)

    def test_online_k2_rejects_invalid_truncation_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                """
training:
  backend: online_k2
  method: sure_k2
  parameter_update_mode: full_parameter
  max_steps: 1
  global_prompt_batch_size: 1
  rollout_micro_batch_size: 1
  teacher_micro_batch_size: 1
  student_micro_batch_size: 1
  max_prompt_tokens: 2
  max_response_tokens: 2
  max_model_length: 4
  sure_alpha: 1.0
  quality_gate:
    warning_truncation_rate: 0.5
    maximum_truncation_rate: 0.2
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "truncation thresholds"):
                load_config(path)

    def test_single_seed_policy_accepts_registered_seed(self) -> None:
        config = load_config("configs/smoke.yaml")
        self.assertEqual(config["project"]["seed"], 42)
        self.assertEqual(config["project"]["registered_seeds"], [42])

    def test_single_seed_policy_rejects_multiple_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                "project:\n  seed: 42\n  seed_policy: single\n  registered_seeds: [42, 43]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_rollout_generation_leaves_context_for_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                "rollout:\n  max_model_length: 4096\n  generation:\n    max_new_tokens: 4096\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "prompt fits"):
                load_config(path)

    def test_benchmark_generation_leaves_context_for_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                "benchmark:\n"
                "  max_model_length: 1024\n"
                "  max_prompt_tokens: 512\n"
                "  generation:\n"
                "    max_new_tokens: 768\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "benchmark.max_prompt_tokens"):
                load_config(path)

    def test_ood_regression_threshold_is_configured(self) -> None:
        config = load_config("configs/sure_k2_24h.yaml")
        regression = config["benchmark"]["ood_regression"]
        self.assertEqual(regression["metric"], "prompt_level_strict_acc")
        self.assertEqual(regression["max_allowed_drop"], 0.02)

    def test_resume_mvp_is_isolated_and_annotates_only_selected_states(self) -> None:
        config = load_config("configs/vfs_weighted_mvp.yaml")

        self.assertEqual(config["project"]["seed"], 42)
        self.assertEqual(config["paths"]["artifact_dir"], "artifacts/resume_mvp")
        self.assertEqual(config["data"]["counts"]["train"], 6000)
        self.assertEqual(config["rollout"]["max_prompts"], 3000)
        self.assertEqual(config["rollout"]["generation"]["num_samples"], 2)
        self.assertEqual(config["rollout"]["max_model_length"], 8192)
        self.assertEqual(config["rollout"]["generation"]["max_new_tokens"], 4096)
        self.assertEqual(config["annotation_selection"]["strategy"], "vfs")
        self.assertEqual(config["annotation_selection"]["budget_ratio"], 0.5)
        self.assertEqual(
            config["teacher"]["input_path"],
            "artifacts/resume_mvp/data/annotation_selection/"
            "vfs_weighted_b50/round_0/selected.parquet",
        )
        self.assertNotIn("dense_b100", config["teacher"]["input_path"])
        self.assertEqual(config["training"]["method"], "weighted_opd")
        self.assertIsNone(config["training"]["max_records"])
        self.assertEqual(config["training"]["max_steps"], 300)
        self.assertEqual(config["project"]["registered_seeds"], [42])
        self.assertNotEqual(config["models"]["teacher"]["revision"], "main")

    def test_resume_mvp_sft_reuses_data_but_has_separate_outputs(self) -> None:
        config = load_config("configs/resume_mvp_sft.yaml")

        self.assertEqual(config["training"]["method"], "sft")
        self.assertEqual(config["training"]["max_records"], 3000)
        self.assertEqual(
            config["training"]["input_path"],
            "artifacts/resume_mvp/data/contamination/train_clean.parquet",
        )
        self.assertEqual(
            config["training"]["output_dir"],
            "artifacts/resume_mvp/checkpoints/sft_seed42",
        )

    def test_resume_mvp_ungated_reuses_teacher_annotations(self) -> None:
        weighted = load_config("configs/vfs_weighted_mvp.yaml")
        ungated = load_config("configs/vfs_ungated_mvp.yaml")

        self.assertEqual(
            ungated["weighting"]["verifier"],
            {
                "pass": 1.0,
                "unknown": 1.0,
                "fail": 1.0,
            },
        )
        self.assertEqual(ungated["project"]["seed"], 42)
        self.assertEqual(ungated["teacher"]["input_path"], weighted["teacher"]["input_path"])
        self.assertEqual(
            ungated["training_view"]["annotation_path"],
            weighted["training_view"]["annotation_path"],
        )
        self.assertNotEqual(
            ungated["training_view"]["output_name"],
            weighted["training_view"]["output_name"],
        )
        self.assertNotEqual(
            ungated["training"]["output_dir"],
            weighted["training"]["output_dir"],
        )

    def test_resume_mvp_vanilla_reuses_selected_teacher_annotations(self) -> None:
        weighted = load_config("configs/vfs_weighted_mvp.yaml")
        vanilla = load_config("configs/vfs_vanilla_mvp.yaml")

        self.assertEqual(vanilla["training"]["method"], "vanilla_opd")
        self.assertEqual(vanilla["project"]["seed"], 42)
        self.assertEqual(vanilla["teacher"]["input_path"], weighted["teacher"]["input_path"])
        self.assertEqual(
            vanilla["training_view"]["annotation_path"],
            weighted["training_view"]["annotation_path"],
        )
        self.assertEqual(
            vanilla["training_view"]["selection_path"],
            weighted["training_view"]["selection_path"],
        )
        self.assertNotEqual(
            vanilla["training_view"]["output_name"],
            weighted["training_view"]["output_name"],
        )

    def test_dense_vanilla_mvp_trains_all_round_zero_rollouts_once(self) -> None:
        config = load_config("configs/dense_vanilla_mvp.yaml")

        self.assertEqual(
            config["teacher"]["input_path"],
            "artifacts/resume_mvp/data/rollouts/round_0/rollouts.parquet",
        )
        self.assertEqual(
            config["teacher"]["reuse_annotation_paths"],
            ["artifacts/resume_mvp/data/annotations/vfs_weighted_b50/round_0/teacher.parquet"],
        )
        self.assertIsNone(config["training_view"]["selection_path"])
        self.assertEqual(config["training"]["method"], "vanilla_opd")
        self.assertFalse(config["training"]["qlora"])
        self.assertEqual(config["training"]["learning_rate"], 0.000002)
        self.assertIsNone(config["training"]["max_records"])
        self.assertEqual(config["training"]["max_steps"], 375)
        self.assertEqual(config["training"]["gradient_accumulation_steps"], 16)
        self.assertEqual(config["training"]["early_stopping"]["eval_steps"], 375)
        self.assertEqual(config["training"]["early_stopping"]["patience"], 3)
        self.assertEqual(
            config["evaluation"]["suites"]["regression"]["input_path"],
            "artifacts/resume_mvp/data/curated/validation.parquet",
        )
        self.assertEqual(
            config["evaluation"]["suites"]["math500"]["input_path"],
            "artifacts/resume_mvp/data/eval/math500.parquet",
        )

    def test_dense_vanilla_lora_is_a_separate_fallback(self) -> None:
        config = load_config("configs/dense_vanilla_lora_mvp.yaml")
        self.assertFalse(config["training"]["qlora"])
        self.assertEqual(config["training"]["parameter_update_mode"], "lora")
        self.assertEqual(config["training"]["method"], "vanilla_opd")
        self.assertNotEqual(
            config["training"]["output_dir"],
            load_config("configs/dense_vanilla_mvp.yaml")["training"]["output_dir"],
        )
        self.assertNotEqual(
            config["training_view"]["output_name"],
            load_config("configs/dense_vanilla_mvp.yaml")["training_view"]["output_name"],
        )

    def test_dense_vanilla_qlora_is_explicitly_quantized(self) -> None:
        config = load_config("configs/dense_vanilla_qlora_mvp.yaml")
        self.assertEqual(config["training"]["parameter_update_mode"], "qlora")
        self.assertTrue(config["training"]["qlora"])
        self.assertNotEqual(
            config["training"]["output_dir"],
            load_config("configs/dense_vanilla_lora_mvp.yaml")["training"]["output_dir"],
        )

    def test_full_parameter_qwen_smoke_disables_qlora(self) -> None:
        config = load_config("configs/qwen_full_parameter_smoke.yaml")
        self.assertFalse(config["training"]["qlora"])
        self.assertEqual(config["training"]["method"], "vanilla_opd")
        self.assertEqual(
            config["training"]["output_dir"],
            "artifacts/qwen_gpu_smoke/checkpoints/full_parameter_seed42",
        )


if __name__ == "__main__":
    unittest.main()
