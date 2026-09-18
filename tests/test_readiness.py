from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from opd.artifacts import build_manifest, save_manifest
from opd.readiness import (
    ReadinessError,
    _check_disk_gate,
    _check_manifest,
    _hardware_is_target,
)


class ReadinessTest(unittest.TestCase):
    def test_manifest_gate_rejects_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "data.parquet"
            artifact.write_bytes(b"stable")
            manifest_path = root / "manifest.json"
            manifest = build_manifest(
                artifact_type="fixture",
                stage="test",
                config={"revision": "old"},
                files=[artifact],
                record_count=1,
                success_count=1,
            )
            save_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ReadinessError, "config hash mismatch"):
                _check_manifest(manifest_path, expected_config_hash="new")

    def test_manifest_gate_rejects_file_checksum_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "data.parquet"
            artifact.write_bytes(b"before")
            manifest_path = root / "manifest.json"
            save_manifest(
                manifest_path,
                build_manifest(
                    artifact_type="fixture",
                    stage="test",
                    config={"revision": "fixed"},
                    files=[artifact],
                    record_count=1,
                    success_count=1,
                ),
            )
            artifact.write_bytes(b"after")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                _check_manifest(manifest_path)

    def test_hardware_gate_requires_target_single_gpu(self) -> None:
        target = {
            "gpu_count": 1,
            "device_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "total_memory_gib": 95.8,
            "compute_capability": [12, 0],
            "compiled_architectures": ["sm_120"],
            "torch_version": "2.7.1+cu128",
            "torch_cuda": "12.8",
        }
        self.assertTrue(_hardware_is_target(target))
        self.assertFalse(_hardware_is_target({**target, "gpu_count": 2}))
        self.assertFalse(_hardware_is_target({**target, "device_name": "NVIDIA A800 80GB"}))

    def test_disk_gate_rejects_insufficient_peak_space(self) -> None:
        config = {
            "paths": {"artifact_dir": "artifacts"},
            "training": {
                "max_steps": 55,
                "milestone_steps": [34],
            },
            "checkpointing": {
                "minimum_free_disk_gib": 80,
                "reserve_artifact_gib": 28,
                "safety_factor": 1.1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            config["paths"]["artifact_dir"] = directory
            plan_path = Path(directory) / "pilot/checkpoint/checkpoint_storage_plan.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text('{"parameter_count": 1700000000, "passed": true}\n')
            pilot = {
                "output_dir": str(plan_path.parent),
            }
            with patch(
                "opd.readiness.shutil.disk_usage",
                return_value=SimpleNamespace(free=1 * 1024**3),
            ):
                with self.assertRaisesRegex(ReadinessError, "insufficient"):
                    _check_disk_gate(config, pilot)


if __name__ == "__main__":
    unittest.main()
