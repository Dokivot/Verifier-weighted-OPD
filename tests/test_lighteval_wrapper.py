from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opd.config import load_config
from opd.evaluation.lighteval import run_lighteval
from opd.tableio import read_json, write_json


class LightEvalWrapperTest(unittest.TestCase):
    def test_wrapper_records_command_metrics_and_manifest(self) -> None:
        config = load_config("configs/smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"

            def fake_run(command: list[str], *, check: bool) -> None:
                self.assertTrue(check)
                write_json(output / "results.json", {"ok": True, "command": command})

            with (
                patch("opd.evaluation.lighteval.shutil.which", return_value="/usr/bin/lighteval"),
                patch("opd.evaluation.lighteval.subprocess.run", side_effect=fake_run),
                patch("opd.artifacts.current_git_commit", return_value="test-commit"),
            ):
                manifest_path = run_lighteval(
                    config,
                    checkpoint="remote/model",
                    output_dir=output,
                    task_names=["math500"],
                )
            manifest = read_json(manifest_path)
            self.assertEqual(manifest["metadata"]["task_names"], ["math500"])
            command = read_json(output / "command.json")["command"]
            self.assertIn("model_name=remote/model", command[2])
            self.assertNotIn("pretrained=", command[2])
            self.assertIn("--use-chat-template", command)
            self.assertIn("--save-details", command)
            self.assertNotIn("--max-samples", command)
            custom_index = command.index("--custom-tasks")
            self.assertEqual(command[custom_index + 1], "opd.evaluation.lighteval_tasks")
            self.assertTrue((output / "job_metrics.json").exists())
            self.assertTrue((output / "command.json").exists())

    def test_ifeval_uses_lighteval_extended_task_discovery(self) -> None:
        config = load_config("configs/base.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"

            def fake_run(command: list[str], *, check: bool) -> None:
                self.assertTrue(check)
                write_json(output / "results.json", {"ok": True})

            with (
                patch("opd.evaluation.lighteval.shutil.which", return_value="lighteval"),
                patch("opd.evaluation.lighteval.subprocess.run", side_effect=fake_run),
                patch("opd.artifacts.current_git_commit", return_value="test-commit"),
            ):
                run_lighteval(
                    config,
                    checkpoint="remote/model",
                    output_dir=output,
                    task_names=["ifeval"],
                )
            command = read_json(output / "command.json")["command"]
            self.assertIn("extended|ifeval|0|0", command)
            self.assertNotIn("--custom-tasks", command)

    def test_math_and_ifeval_share_one_custom_task_module(self) -> None:
        config = load_config("configs/base.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"

            def fake_run(command: list[str], *, check: bool) -> None:
                self.assertTrue(check)
                write_json(output / "results.json", {"ok": True})

            with (
                patch("opd.evaluation.lighteval.shutil.which", return_value="lighteval"),
                patch("opd.evaluation.lighteval.subprocess.run", side_effect=fake_run),
                patch("opd.artifacts.current_git_commit", return_value="test-commit"),
            ):
                run_lighteval(
                    config,
                    checkpoint="remote/model",
                    output_dir=output,
                    task_names=["math500", "aime2024", "ifeval"],
                )
            command = read_json(output / "command.json")["command"]
            self.assertIn(
                "lighteval|math_500|0|0,lighteval|aime24|0|0,extended|ifeval|0|0",
                command,
            )
            self.assertEqual(command.count("--custom-tasks"), 1)
            custom_index = command.index("--custom-tasks")
            self.assertEqual(command[custom_index + 1], "opd.evaluation.lighteval_tasks")

    def test_max_samples_is_forwarded_to_lighteval(self) -> None:
        config = load_config("configs/qwen_gpu_smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"

            def fake_run(command: list[str], *, check: bool) -> None:
                self.assertTrue(check)
                write_json(output / "results.json", {"ok": True})

            with (
                patch("opd.evaluation.lighteval.shutil.which", return_value="lighteval"),
                patch("opd.evaluation.lighteval.subprocess.run", side_effect=fake_run),
                patch("opd.artifacts.current_git_commit", return_value="test-commit"),
            ):
                run_lighteval(
                    config,
                    checkpoint="remote/model",
                    output_dir=output,
                    task_names=["math500"],
                )
            command = read_json(output / "command.json")["command"]
            self.assertIn("max_model_length=8192", command[2])
            max_samples_index = command.index("--max-samples")
            self.assertEqual(command[max_samples_index + 1], "2")


if __name__ == "__main__":
    unittest.main()
