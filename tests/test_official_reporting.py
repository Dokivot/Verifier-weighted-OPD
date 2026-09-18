from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.artifacts import build_manifest, save_manifest
from opd.config import load_config
from opd.reporting.official import load_official_scores
from opd.tableio import write_json


class OfficialReportingTest(unittest.TestCase):
    def _run(self, root: Path, *, include_metric: bool) -> Path:
        run_dir = root / "base_math500"
        result_path = run_dir / "results" / "results.json"
        command_path = run_dir / "command.json"
        metrics_path = run_dir / "job_metrics.json"
        result = {
            "results": {
                "lighteval|math_500|0|0": (
                    {"math_pass@1:1_samples": 0.42, "math_pass@1:1_samples_stderr": 0.02}
                    if include_metric
                    else {"math_pass@1:4_samples": 0.50}
                )
            }
        }
        write_json(result_path, result)
        write_json(command_path, {"command": ["lighteval"]})
        write_json(metrics_path, {"job_type": "lighteval"})
        save_manifest(
            run_dir / "manifest.json",
            build_manifest(
                artifact_type="benchmark_run",
                stage="benchmark.lighteval",
                config={"test": True},
                files=[result_path, command_path, metrics_path],
                record_count=0,
                success_count=1,
                metadata={
                    "checkpoint": "student",
                    "task_names": ["math500"],
                    "task_identifiers": "lighteval|math_500|0|0",
                    "generation_protocol": {"enable_thinking": False},
                },
            ),
        )
        return run_dir / "manifest.json"

    def test_parses_math500_primary_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(load_config("configs/base.yaml"))
            config["benchmark"]["base_math_output_dir"] = str(root / "base_math500")
            manifest_path = self._run(root, include_metric=True)
            scores = load_official_scores(config, [manifest_path])
            self.assertEqual(scores[0]["metric"], "math_pass@1:1_samples")
            self.assertAlmostEqual(scores[0]["score"], 0.42)
            self.assertEqual(scores[0]["model"], "Base")

    def test_missing_primary_metric_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config("configs/base.yaml")
            manifest_path = self._run(root, include_metric=False)
            with self.assertRaisesRegex(ValueError, "was not found"):
                load_official_scores(config, [manifest_path])


if __name__ == "__main__":
    unittest.main()
