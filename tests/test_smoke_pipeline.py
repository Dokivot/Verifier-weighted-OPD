from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.artifacts import build_manifest, save_manifest
from opd.config import load_config
from opd.data.contamination import audit_contamination
from opd.data.prepare import prepare_dataset
from opd.evaluation.runner import evaluate
from opd.reporting.build import build_report
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


if __name__ == "__main__":
    unittest.main()
