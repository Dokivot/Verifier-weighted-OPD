from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.artifacts import build_manifest, save_manifest
from opd.reporting.analysis import build_cost_pareto, build_failure_analysis
from opd.tableio import write_json, write_records


class ReportingAnalysisTest(unittest.TestCase):
    def test_failure_analysis_and_pareto_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            write_records(
                predictions,
                [
                    {"sample_id": "a", "score": 1, "status": "pass", "response_tokens": 2},
                    {
                        "sample_id": "b",
                        "score": 0,
                        "status": "fail",
                        "subject": "algebra",
                        "difficulty": "hard",
                        "response_tokens": 10,
                        "response": "wrong",
                    },
                ],
            )
            config = {"project": {"seed": 42}, "report": {}}
            predictions_manifest = build_manifest(
                artifact_type="evaluation_run",
                stage="evaluate",
                config=config,
                files=[predictions],
                record_count=2,
                success_count=2,
            )
            save_manifest(root / "manifest.json", predictions_manifest)
            failure_report = build_failure_analysis(config, predictions, root / "failures.json")
            self.assertEqual(failure_report["failures"], 1)
            self.assertTrue((root / "failures.md").exists())

            base_summary = root / "base.json"
            better_summary = root / "better.json"
            write_json(base_summary, {"run_id": "base", "accuracy": 0.5})
            write_json(better_summary, {"run_id": "better", "accuracy": 0.6})
            base_metrics = root / "base_metrics.json"
            better_metrics = root / "better_metrics.json"
            write_json(base_metrics, {"estimated_gpu_hours": 1, "teacher_tokens": 0})
            write_json(better_metrics, {"estimated_gpu_hours": 2, "teacher_tokens": 100})
            base_manifest = build_manifest(
                artifact_type="evaluation_run",
                stage="evaluate",
                config=config,
                files=[base_summary, base_metrics],
                record_count=1,
                success_count=1,
            )
            better_manifest = build_manifest(
                artifact_type="evaluation_run",
                stage="evaluate",
                config=config,
                files=[better_summary, better_metrics],
                record_count=1,
                success_count=1,
            )
            save_manifest(base_summary.with_suffix(".manifest.json"), base_manifest)
            save_manifest(better_summary.with_suffix(".manifest.json"), better_manifest)
            config = {
                "project": {"seed": 42},
                "report": {
                    "runs": [
                        {
                            "name": "Base",
                            "summary_path": str(base_summary),
                            "job_metric_paths": [str(base_metrics)],
                        },
                        {
                            "name": "Weighted",
                            "summary_path": str(better_summary),
                            "job_metric_paths": [str(better_metrics)],
                        },
                    ]
                },
            }
            pareto = build_cost_pareto(config, root / "pareto")
            self.assertEqual(len(pareto["points"]), 2)
            self.assertTrue((root / "pareto/quality_cost.svg").exists())


if __name__ == "__main__":
    unittest.main()
