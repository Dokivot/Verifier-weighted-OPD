from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.config import load_config
from opd.exceptions import ConfigurationError


class ConfigTest(unittest.TestCase):
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
