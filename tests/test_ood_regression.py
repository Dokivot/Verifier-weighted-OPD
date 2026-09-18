from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.artifacts import build_manifest, save_manifest
from opd.config import load_config
from opd.reporting.build import build_report
from opd.reporting.ood import build_ood_regression
from opd.tableio import read_json, write_json


class OODRegressionTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        name: str,
        score: float,
        *,
        metric: str = "prompt_level_strict_acc",
    ) -> Path:
        run_dir = root / name
        result_path = run_dir / "results" / "results.json"
        command_path = run_dir / "command.json"
        metrics_path = run_dir / "job_metrics.json"
        write_json(
            result_path,
            {"results": {"extended|ifeval|0|0": {metric: score}}},
        )
        write_json(command_path, {"command": ["lighteval"], "task_names": ["ifeval"]})
        write_json(metrics_path, {"job_type": "lighteval", "wall_time_seconds": 1})
        manifest = build_manifest(
            artifact_type="benchmark_run",
            stage="benchmark.lighteval",
            config={"test": True},
            files=[result_path, command_path, metrics_path],
            record_count=0,
            success_count=1,
            metadata={
                "run_id": name,
                "task_names": ["ifeval"],
                "task_identifiers": "extended|ifeval|0|0",
                "generation_protocol": {
                    "max_new_tokens": 8192,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": -1,
                    "stop_strategy": "chat_template_eos",
                },
            },
        )
        save_manifest(run_dir / "manifest.json", manifest)
        return run_dir

    def _config(self) -> dict:
        config = deepcopy(load_config("configs/base.yaml"))
        config["project"]["seed"] = 42
        config["benchmark"]["ood_regression"] = {
            "metric": "prompt_level_strict_acc",
            "max_allowed_drop": 0.02,
        }
        return config

    def test_builds_task_and_overall_delta_with_protocol_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config()
            baseline = self._run(root, "base", 0.75)
            candidate = self._run(root, "sure_k2", 0.74)
            output = build_ood_regression(
                config,
                baseline_dir=baseline,
                candidate_dirs={"sure_k2": candidate},
                output_path=root / "report" / "ood_regression.json",
            )
            report = read_json(output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["metric"], "prompt_level_strict_acc")
            self.assertAlmostEqual(report["deltas"]["sure_k2"]["overall_delta"], -0.01)
            self.assertTrue(report["gate"]["passed"])
            self.assertTrue((root / "report/ood_regression.manifest.json").exists())

    def test_regression_writes_diagnostic_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config()
            baseline = self._run(root, "base", 0.75)
            candidate = self._run(root, "sure_k2", 0.70)
            output = root / "report" / "ood_regression.json"
            with self.assertRaisesRegex(ValueError, "OOD regression gate failed"):
                build_ood_regression(
                    config,
                    baseline_dir=baseline,
                    candidate_dirs={"sure_k2": candidate},
                    output_path=output,
                )
            report = read_json(output)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["gate"]["passed"])
            self.assertTrue((root / "report/ood_regression.manifest.json").exists())

    def test_missing_metric_is_not_silently_treated_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config()
            baseline = self._run(root, "base", 0.75)
            candidate = self._run(root, "sure_k2", 0.74, metric="different_metric")
            with self.assertRaisesRegex(ValueError, "was not found"):
                build_ood_regression(
                    config,
                    baseline_dir=baseline,
                    candidate_dirs={"sure_k2": candidate},
                    output_path=root / "report" / "ood_regression.json",
                )

    def test_final_report_includes_passed_ood_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config()
            baseline = self._run(root, "base", 0.75)
            candidate = self._run(root, "sure_k2", 0.75)
            regression_path = root / "report" / "ood_regression.json"
            build_ood_regression(
                config,
                baseline_dir=baseline,
                candidate_dirs={"sure_k2": candidate},
                output_path=regression_path,
            )
            config["report"]["output_dir"] = str(root / "report")
            config["report"]["ood_regression_path"] = str(regression_path)
            config["benchmark"]["base_output_dir"] = str(baseline)
            config["benchmark"]["candidate_output_dir"] = str(candidate)
            config["report"]["official_benchmark_manifests"] = [
                str(baseline / "manifest.json"),
                str(candidate / "manifest.json"),
            ]
            report_path = build_report(config, "experiment")
            self.assertIn("## OOD Regression", report_path.read_text(encoding="utf-8"))
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Official Benchmark", report_text)
            self.assertIn("0.7500", report_text)
            self.assertIn("0.7500", report_text)
            report_json = read_json(root / "report/experiment.json")
            self.assertEqual(report_json["ood_regression"]["report"]["status"], "passed")
            self.assertEqual(len(report_json["official_benchmark"]), 2)


if __name__ == "__main__":
    unittest.main()
