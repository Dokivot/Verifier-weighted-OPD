from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.config import load_config
from opd.data.annotation_selection import select_annotation_rollouts
from opd.data.contamination import audit_contamination
from opd.data.prepare import prepare_dataset
from opd.rollout.pipeline import generate_rollouts
from opd.tableio import read_json, read_records
from opd.teacher.pipeline import annotate_rollouts
from opd.training.views import build_training_view
from opd.verifier.pipeline import verify_rollouts


class AnnotationSelectionTest(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        config = deepcopy(load_config("configs/smoke.yaml"))
        data_dir = root / "data"
        config["paths"]["artifact_dir"] = str(root)
        config["paths"]["data_dir"] = str(data_dir)
        config["data"]["input_path"] = str(Path("tests/fixtures/math_smoke.jsonl").resolve())
        config["contamination"]["train_path"] = str(data_dir / "curated/train.jsonl")
        config["contamination"]["eval_paths"] = [str(data_dir / "curated/validation.jsonl")]
        config["rollout"]["generation"]["num_samples"] = 2
        config["annotation_selection"] = {
            "enabled": True,
            "strategy": "vfs",
            "budget_ratio": 0.5,
            "status_priority": ["pass", "unknown", "fail"],
            "group_priority": ["boundary", "uncertain", "solved", "failed"],
            "stratify_by": ["subject", "difficulty"],
        }
        return config

    def test_vfs_is_deterministic_and_never_exceeds_teacher_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            prepare_dataset(config)
            audit_contamination(config)
            generate_rollouts(config, round_id=0)
            verify_rollouts(config, round_id=0)

            selected_path = select_annotation_rollouts(config, round_id=0)
            first_selected = read_records(selected_path)
            first_decisions = read_records(
                root / "data/annotation_selection/round_0/decisions.jsonl"
            )
            report = read_json(root / "data/annotation_selection/round_0/selection_report.json")

            self.assertGreater(len(first_selected), 0)
            self.assertLessEqual(
                report["selected_estimated_teacher_tokens"],
                report["teacher_token_budget"],
            )
            self.assertEqual(report["candidate_records"], 32)
            self.assertEqual(len({row["sample_id"] for row in first_selected}), 16)
            self.assertTrue(
                all(row["group_type"] in {"solved", "failed"} for row in first_decisions)
            )

            select_annotation_rollouts(config, round_id=0)
            self.assertEqual(first_selected, read_records(selected_path))

    def test_random_budget_uses_the_same_budget_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            config["annotation_selection"]["strategy"] = "random_budget"
            prepare_dataset(config)
            audit_contamination(config)
            generate_rollouts(config, round_id=0)
            verify_rollouts(config, round_id=0)

            select_annotation_rollouts(config, round_id=0)
            report = read_json(root / "data/annotation_selection/round_0/selection_report.json")

            self.assertEqual(
                report["teacher_token_budget"],
                int(report["dense_estimated_teacher_tokens"] * 0.5),
            )
            self.assertLessEqual(
                report["selected_estimated_teacher_tokens"],
                report["teacher_token_budget"],
            )

    def test_training_view_reuses_dense_annotations_for_selected_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            prepare_dataset(config)
            audit_contamination(config)
            rollout_path = generate_rollouts(config, round_id=0)
            verify_rollouts(config, round_id=0)
            selected_path = select_annotation_rollouts(config, round_id=0)

            dense_annotation_dir = root / "data/annotations/dense_b100/round_0"
            config["teacher"]["input_path"] = str(rollout_path)
            config["teacher"]["output_dir"] = str(dense_annotation_dir)
            annotation_path = annotate_rollouts(config, round_id=0)
            config["training_view"] = {
                "annotation_path": str(annotation_path),
                "selection_path": str(selected_path),
                "output_dir": str(root / "data/training_views/round_0"),
                "output_name": "vfs_b50",
            }

            view_path = build_training_view(config, round_id=0, method="vanilla_opd")

            self.assertEqual(len(read_records(view_path)), len(read_records(selected_path)))
            manifest = read_json(view_path.with_suffix(".manifest.json"))
            self.assertEqual(manifest["metadata"]["selection_path"], str(selected_path))


if __name__ == "__main__":
    unittest.main()
