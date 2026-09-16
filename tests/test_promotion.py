from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.artifacts import build_manifest, save_manifest
from opd.checkpoints.promotion import evaluate_promotion
from opd.config import load_config
from opd.tableio import write_json


class PromotionTest(unittest.TestCase):
    def test_checkpoint_is_promoted_when_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(load_config("configs/smoke.yaml"))
            training_output = root / "training"
            checkpoint = training_output / "checkpoint.json"
            write_json(checkpoint, {"mock": True})
            write_json(
                training_output / "training_summary.json",
                {"history": [{"step": 1, "loss": 1.0}]},
            )
            upstream = root / "upstream.json"
            write_json(upstream, {"value": 1})
            manifest = build_manifest(
                artifact_type="checkpoint",
                stage="train",
                config=config,
                files=[checkpoint, training_output / "training_summary.json"],
                record_count=1,
                success_count=1,
                upstream_artifact_ids=["upstream-id"],
            )
            save_manifest(training_output / "manifest.json", manifest)
            config["training"]["output_dir"] = str(training_output)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "samples": 10,
                "accuracy": 0.5,
                "mean_response_tokens": 20,
                "status_counts": {"unknown": 0},
            }
            write_json(baseline, {**common, "run_id": "base"})
            write_json(candidate, {**common, "run_id": "candidate", "accuracy": 0.6})
            baseline_manifest = build_manifest(
                artifact_type="evaluation_run",
                stage="evaluate",
                config=config,
                files=[baseline],
                record_count=10,
                success_count=10,
            )
            candidate_manifest = build_manifest(
                artifact_type="evaluation_run",
                stage="evaluate",
                config=config,
                files=[candidate],
                record_count=10,
                success_count=10,
            )
            save_manifest(baseline.with_suffix(".manifest.json"), baseline_manifest)
            save_manifest(candidate.with_suffix(".manifest.json"), candidate_manifest)
            decision = evaluate_promotion(
                config,
                baseline_summary_path=baseline,
                candidate_summary_path=candidate,
                checkpoint_path=checkpoint,
                output_path=root / "promotion/decision.json",
            )
            self.assertTrue(decision["promotable"])
            self.assertTrue((root / "promotion/PROMOTABLE").exists())


if __name__ == "__main__":
    unittest.main()
