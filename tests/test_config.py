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


if __name__ == "__main__":
    unittest.main()
