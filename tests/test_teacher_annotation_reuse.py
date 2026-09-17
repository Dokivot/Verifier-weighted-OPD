from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from opd.artifacts import build_manifest, save_manifest
from opd.config import load_config
from opd.data.contamination import audit_contamination
from opd.data.prepare import prepare_dataset
from opd.rollout.pipeline import generate_rollouts
from opd.tableio import read_json, read_records, write_records
from opd.teacher.mock import MockTeacherAnnotator
from opd.teacher.pipeline import annotate_rollouts


class CountingTeacher:
    def __init__(self, *, top_k: int) -> None:
        self.delegate = MockTeacherAnnotator(top_k=top_k)
        self.model_name = self.delegate.model_name
        self.model_revision = self.delegate.model_revision
        self.tokenizer_revision = self.delegate.tokenizer.revision
        self.tokenizer_fingerprint = self.delegate.tokenizer_fingerprint
        self.calls = 0

    def annotate(self, prompt: str, response: str) -> dict[str, object]:
        self.calls += 1
        return self.delegate.annotate(prompt, response)


class TeacherAnnotationReuseTest(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        config = deepcopy(load_config("configs/smoke.yaml"))
        data_dir = root / "data"
        config["paths"]["artifact_dir"] = str(root)
        config["paths"]["data_dir"] = str(data_dir)
        config["data"]["input_path"] = str(Path("tests/fixtures/math_smoke.jsonl").resolve())
        config["contamination"]["train_path"] = str(data_dir / "curated/train.jsonl")
        config["contamination"]["eval_paths"] = [str(data_dir / "curated/validation.jsonl")]
        return config

    def _prepare_seed_annotations(
        self, root: Path, config: dict, *, count: int
    ) -> tuple[Path, Path]:
        prepare_dataset(config)
        audit_contamination(config)
        rollout_path = generate_rollouts(config, round_id=0)
        subset_path = root / "data/reuse_input/selected.jsonl"
        subset_rows = read_records(rollout_path)[:count]
        write_records(subset_path, subset_rows)
        subset_manifest = build_manifest(
            artifact_type="annotation_selection",
            stage="test.select",
            config=config,
            files=[subset_path],
            record_count=len(subset_rows),
            success_count=len(subset_rows),
        )
        save_manifest(subset_path.parent / "manifest.json", subset_manifest)
        config["teacher"]["input_path"] = str(subset_path)
        config["teacher"]["output_dir"] = str(root / "data/reuse_annotations")
        seed_path = annotate_rollouts(config, round_id=0)
        return rollout_path, seed_path

    def test_reuses_compatible_records_and_generates_only_missing_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            rollout_path, seed_path = self._prepare_seed_annotations(root, config, count=6)
            rollout_ids = [row["rollout_id"] for row in read_records(rollout_path)]
            teacher = CountingTeacher(top_k=4)
            config["teacher"]["input_path"] = str(rollout_path)
            config["teacher"]["output_dir"] = str(root / "data/dense_annotations")
            config["teacher"]["reuse_annotation_paths"] = [str(seed_path)]

            with patch("opd.teacher.pipeline._annotator", return_value=teacher):
                output_path = annotate_rollouts(config, round_id=0)

            output_rows = read_records(output_path)
            output_ids = [row["rollout_id"] for row in output_rows]
            manifest = read_json(output_path.parent / "manifest.json")
            self.assertEqual(output_ids, rollout_ids)
            self.assertEqual(len(output_ids), len(set(output_ids)))
            self.assertEqual(teacher.calls, 10)
            self.assertEqual(manifest["metadata"]["reused_annotation_records"], 6)
            self.assertEqual(manifest["metadata"]["generated_records"], 10)
            self.assertEqual(manifest["record_count"], 16)

    def test_rejects_incompatible_reusable_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            rollout_path, seed_path = self._prepare_seed_annotations(root, config, count=6)
            seed_rows = read_records(seed_path)
            seed_rows[0]["teacher_revision"] = "incompatible-revision"
            write_records(seed_path, seed_rows)
            replacement_manifest = build_manifest(
                artifact_type="teacher_annotation",
                stage="teacher.annotate",
                config=config,
                files=[seed_path],
                record_count=len(seed_rows),
                success_count=len(seed_rows),
            )
            save_manifest(seed_path.parent / "manifest.json", replacement_manifest)
            config["teacher"]["input_path"] = str(rollout_path)
            config["teacher"]["output_dir"] = str(root / "data/dense_annotations")
            config["teacher"]["reuse_annotation_paths"] = [str(seed_path)]

            with self.assertRaisesRegex(ValueError, "incompatible-revision"):
                annotate_rollouts(config, round_id=0)


if __name__ == "__main__":
    unittest.main()
