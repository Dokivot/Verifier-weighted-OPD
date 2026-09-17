from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from opd.artifacts import build_manifest, save_manifest
from opd.config import load_config
from opd.data.annotation_selection import select_annotation_rollouts
from opd.data.contamination import audit_contamination
from opd.data.prepare import prepare_dataset
from opd.evaluation.runner import evaluate
from opd.reporting.build import build_report
from opd.rollout.base import Generation
from opd.rollout.pipeline import generate_rollouts
from opd.tableio import read_json, read_records, write_records
from opd.teacher.pipeline import annotate_rollouts
from opd.training.pipeline import train
from opd.training.views import build_training_view
from opd.verifier.pipeline import verify_rollouts


class SmokePipelineTest(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        config = deepcopy(load_config("configs/smoke.yaml"))
        data_dir = root / "data"
        config["paths"]["artifact_dir"] = str(root)
        config["paths"]["data_dir"] = str(data_dir)
        config["paths"]["checkpoint_dir"] = str(root / "checkpoints")
        config["paths"]["report_dir"] = str(root / "reports")
        config["data"]["input_path"] = str(Path("tests/fixtures/math_smoke.jsonl").resolve())
        config["contamination"]["train_path"] = str(data_dir / "curated/train.jsonl")
        config["contamination"]["eval_paths"] = [str(data_dir / "curated/validation.jsonl")]
        config["training"]["input_path"] = str(
            data_dir / "training_views/round_0/weighted_opd.jsonl"
        )
        config["training"]["validation_path"] = str(data_dir / "curated/validation.jsonl")
        config["training"]["output_dir"] = str(root / "checkpoints/weighted")
        config["evaluation"]["output_dir"] = str(root / "evaluation/weighted")
        config["evaluation"]["suites"]["smoke"]["input_path"] = str(
            data_dir / "curated/validation.jsonl"
        )
        summary_path = root / "evaluation/weighted/smoke/summary.json"
        config["report"]["output_dir"] = str(root / "reports")
        config["report"]["summary_paths"] = [str(summary_path)]
        config["report"]["training_runs"][0]["summary_path"] = str(
            root / "checkpoints/weighted/training_summary.json"
        )
        return config

    def test_end_to_end_and_shard_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            outputs = prepare_dataset(config)
            self.assertEqual(len(read_records(outputs["train"])), 16)
            contamination = audit_contamination(config)
            self.assertEqual(contamination["match_count"], 0)

            rollout_path = generate_rollouts(config, round_id=0)
            first_rows = read_records(rollout_path)
            self.assertEqual(len(first_rows), 16)
            generate_rollouts(config, round_id=0)
            rollout_manifest = read_json(root / "data/rollouts/round_0/manifest.json")
            self.assertEqual(rollout_manifest["metadata"]["reused_shards"], 4)
            self.assertEqual(read_records(rollout_path), first_rows)

            verify_rollouts(config, round_id=0)
            select_annotation_rollouts(config, round_id=0)
            annotate_rollouts(config, round_id=0)
            annotate_rollouts(config, round_id=0)
            teacher_manifest = read_json(root / "data/annotations/round_0/manifest.json")
            self.assertEqual(teacher_manifest["metadata"]["reused_shards"], 4)

            view_path = build_training_view(config, round_id=0, method="weighted_opd")
            self.assertEqual(len(read_records(view_path)), 16)
            self.assertTrue(train(config).exists())
            summary_path = evaluate(config, suite="smoke")
            self.assertEqual(read_json(summary_path)["accuracy"], 1.0)
            self.assertTrue(build_report(config, "smoke").exists())

    def test_training_view_rejects_tokenizer_vocabulary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            prepare_dataset(config)
            audit_contamination(config)
            generate_rollouts(config, round_id=0)
            verify_rollouts(config, round_id=0)
            select_annotation_rollouts(config, round_id=0)
            annotation_path = annotate_rollouts(config, round_id=0)
            rows = read_records(annotation_path)
            rows[0]["tokenizer_fingerprint"] = "different-vocabulary"
            write_records(annotation_path, rows)
            manifest = build_manifest(
                artifact_type="teacher_annotation",
                stage="teacher.annotate",
                config=config,
                files=[annotation_path],
                record_count=len(rows),
                success_count=len(rows),
            )
            save_manifest(root / "data/annotations/round_0/manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Tokenizer vocabulary mismatch"):
                build_training_view(config, round_id=0, method="weighted_opd")

    def test_teacher_annotation_fails_stage_after_recording_errors(self) -> None:
        class FailingTeacher:
            model_name = "failing-teacher"
            model_revision = "failing-v1"
            tokenizer_revision = "mock-tokenizer-v1"
            tokenizer_fingerprint = "mock-tokenizer"

            def annotate(self, prompt: str, response: str) -> dict[str, object]:
                del prompt, response
                raise RuntimeError("synthetic teacher failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            prepare_dataset(config)
            audit_contamination(config)
            generate_rollouts(config, round_id=0)
            verify_rollouts(config, round_id=0)
            select_annotation_rollouts(config, round_id=0)

            with (
                patch("opd.teacher.pipeline._annotator", return_value=FailingTeacher()),
                self.assertRaisesRegex(RuntimeError, "16 failed records"),
            ):
                annotate_rollouts(config, round_id=0)

            manifest = read_json(root / "data/annotations/round_0/manifest.json")
            self.assertEqual(manifest["failure_count"], 16)
            self.assertEqual(manifest["success_count"], 0)

    def test_rollout_truncation_gate_stops_after_minimum_sample(self) -> None:
        class CappedBackend:
            model_name = "capped-student"
            model_revision = "capped-v1"
            tokenizer_revision = "capped-tokenizer-v1"
            tokenizer_fingerprint = "capped-tokenizer"

            def generate(self, prompts: list[str], *, seed: int) -> list[Generation]:
                del seed
                return [
                    Generation(text="unfinished", prompt_tokens=4, response_tokens=3)
                    for _ in prompts
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            config["rollout"]["max_prompts"] = 8
            config["rollout"]["shard_size"] = 4
            config["rollout"]["generation"]["max_new_tokens"] = 3
            config["rollout"]["quality_gate"] = {
                "enabled": True,
                "min_samples": 4,
                "max_truncation_rate": 0.2,
            }
            prepare_dataset(config)
            audit_contamination(config)

            with (
                patch("opd.rollout.pipeline._backend", return_value=CappedBackend()),
                self.assertRaisesRegex(RuntimeError, "100.00% exceeded 20.00%"),
            ):
                generate_rollouts(config, round_id=0)

            quality = read_json(root / "data/rollouts/round_0/quality_gate.json")
            self.assertEqual(quality["status"], "failed")
            self.assertEqual(quality["successful_records"], 4)
            self.assertEqual(quality["truncated_records"], 4)
            self.assertEqual(
                len(list((root / "data/rollouts/round_0/shards").rglob("*.jsonl"))),
                1,
            )

    def test_rollout_limits_the_selected_prompt_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            config["rollout"]["max_prompts"] = 7
            prepare_dataset(config)
            audit_contamination(config)

            rollout_path = generate_rollouts(config, round_id=0)

            self.assertEqual(len(read_records(rollout_path)), 7)
            manifest = read_json(root / "data/rollouts/round_0/manifest.json")
            self.assertEqual(manifest["metadata"]["available_prompt_count"], 16)
            self.assertEqual(manifest["metadata"]["selected_prompt_count"], 7)

    def test_rollout_gate_requires_enough_successful_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            config["rollout"]["max_prompts"] = 3
            config["rollout"]["quality_gate"] = {
                "enabled": True,
                "min_samples": 4,
                "max_truncation_rate": 0.2,
            }
            prepare_dataset(config)
            audit_contamination(config)

            with self.assertRaisesRegex(RuntimeError, "only 3 successful records"):
                generate_rollouts(config, round_id=0)

            quality = read_json(root / "data/rollouts/round_0/quality_gate.json")
            self.assertEqual(quality["status"], "collecting")
            self.assertFalse((root / "data/rollouts/round_0/rollouts.jsonl").exists())
            self.assertFalse((root / "data/rollouts/round_0/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
