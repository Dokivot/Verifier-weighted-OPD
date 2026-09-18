from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.artifacts import build_manifest, save_manifest
from opd.config import load_config
from opd.schemas import RecordStatus, RolloutRecord, VerificationRecord, VerificationStatus
from opd.tableio import read_json, read_records, write_records
from opd.verifier.calibration import calibrate_verifier


class VerifierCalibrationTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[dict, Path, Path]:
        config = deepcopy(load_config("configs/smoke.yaml"))
        config["project"]["seed"] = 42
        config["rollout"]["generation"]["max_new_tokens"] = 8

        rollout_path = root / "rollouts.jsonl"
        verification_path = root / "verification.jsonl"
        rollout_rows: list[dict] = []
        verification_rows: list[dict] = []
        statuses = [
            VerificationStatus.PASS,
            VerificationStatus.FAIL,
            VerificationStatus.UNKNOWN,
        ]
        for index, status in enumerate(statuses):
            for copy_index in range(2):
                rollout_id = f"r-{index}-{copy_index}"
                rollout_rows.append(
                    RolloutRecord(
                        rollout_id=rollout_id,
                        sample_id=f"s-{index}-{copy_index}",
                        round_id=0,
                        candidate_index=copy_index,
                        prompt="Solve 1 + 1.",
                        response=f"answer {index}-{copy_index}",
                        student_model="student",
                        student_revision="student-v1",
                        tokenizer_revision="tokenizer-v1",
                        tokenizer_fingerprint="tokenizer-fingerprint",
                        sampling_config_hash="sampling-v1",
                        seed=42,
                        prompt_tokens=4,
                        response_tokens=2,
                        finish_reason="stop",
                        latency_ms=1,
                    ).model_dump(mode="json")
                )
                verification_rows.append(
                    VerificationRecord(
                        rollout_id=rollout_id,
                        sample_id=f"s-{index}-{copy_index}",
                        verifier_name="math",
                        verifier_version="2",
                        status=status,
                        score=1.0 if status is VerificationStatus.PASS else 0.0,
                        reference_answer="2",
                        extracted_answer="2" if status is VerificationStatus.PASS else None,
                        error_type=(
                            None if status is not VerificationStatus.FAIL else "wrong_answer"
                        ),
                        details={
                            "reason": {
                                VerificationStatus.PASS: "equivalent",
                                VerificationStatus.FAIL: "not_equivalent",
                                VerificationStatus.UNKNOWN: "answer_not_found",
                            }[status],
                            "extraction_confidence": (
                                "high" if status is VerificationStatus.PASS else "none"
                            ),
                        },
                    ).model_dump(mode="json")
                )

        truncated_id = "r-truncated"
        rollout_rows.append(
            RolloutRecord(
                rollout_id=truncated_id,
                sample_id="s-truncated",
                round_id=0,
                candidate_index=0,
                prompt="Solve 1 + 1.",
                response="unfinished reasoning",
                student_model="student",
                student_revision="student-v1",
                tokenizer_revision="tokenizer-v1",
                tokenizer_fingerprint="tokenizer-fingerprint",
                sampling_config_hash="sampling-v1",
                seed=42,
                prompt_tokens=4,
                response_tokens=8,
                finish_reason="length",
                status=RecordStatus.TRUNCATED,
                latency_ms=1,
            ).model_dump(mode="json")
        )
        verification_rows.append(
            VerificationRecord(
                rollout_id=truncated_id,
                sample_id="s-truncated",
                verifier_name="math",
                verifier_version="2",
                status=VerificationStatus.UNKNOWN,
                score=0.0,
                reference_answer="2",
                error_type="answer_not_found",
                truncated=True,
                details={
                    "reason": "answer_not_found",
                    "extraction_confidence": "none",
                },
            ).model_dump(mode="json")
        )
        write_records(rollout_path, rollout_rows)
        write_records(verification_path, verification_rows)
        for path, artifact_type in (
            (rollout_path, "rollout"),
            (verification_path, "verification"),
        ):
            save_manifest(
                path.parent / f"{path.stem}.manifest.json",
                build_manifest(
                    artifact_type=artifact_type,
                    stage=f"test.{artifact_type}",
                    config=config,
                    files=[path],
                    record_count=len(read_records(path)),
                    success_count=len(read_records(path)),
                ),
            )
        return config, verification_path, rollout_path

    def test_queue_preserves_unknown_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, verification_path, rollout_path = self._inputs(root)
            first = calibrate_verifier(
                config,
                verification_path=verification_path,
                rollout_path=rollout_path,
                output_dir=root / "first",
                samples_per_stratum=1,
            )
            calibrate_verifier(
                config,
                verification_path=verification_path,
                rollout_path=rollout_path,
                output_dir=root / "second",
                samples_per_stratum=1,
            )

            first_report = read_json(first)
            queue = read_records(root / "first/review_queue.jsonl")
            second_queue = read_records(root / "second/review_queue.jsonl")
            self.assertEqual(first_report["status"], "awaiting_manual_labels")
            self.assertTrue(first_report["unknown_is_preserved"])
            self.assertEqual(
                {row["stratum"] for row in queue},
                {"pass", "fail", "unknown", "truncated"},
            )
            self.assertEqual(queue, second_queue)
            self.assertEqual(first_report["internal_verifier"]["reviewed_samples"], 0)

    def test_requires_all_labels_and_reports_disagreements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, verification_path, rollout_path = self._inputs(root)
            queue_dir = root / "queue"
            calibrate_verifier(
                config,
                verification_path=verification_path,
                rollout_path=rollout_path,
                output_dir=queue_dir,
                samples_per_stratum=1,
            )
            queue = read_records(queue_dir / "review_queue.jsonl")
            incomplete_labels = root / "incomplete_labels.jsonl"
            write_records(
                incomplete_labels,
                [{"rollout_id": queue[0]["rollout_id"], "human_status": "pass"}],
            )
            with self.assertRaisesRegex(ValueError, "Manual labels are missing"):
                calibrate_verifier(
                    config,
                    verification_path=verification_path,
                    rollout_path=rollout_path,
                    output_dir=root / "incomplete",
                    labels_path=incomplete_labels,
                    samples_per_stratum=1,
                )

            labels_path = root / "labels.jsonl"
            human_status = {
                "pass": "pass",
                "fail": "fail",
                "unknown": "unknown",
                "truncated": "pass",
            }
            write_records(
                labels_path,
                [
                    {
                        "rollout_id": row["rollout_id"],
                        "human_status": human_status[row["stratum"]],
                        "human_error_type": None,
                        "reviewer": "reviewer-1",
                        "notes": "checked",
                    }
                    for row in queue
                ],
            )
            report_path = calibrate_verifier(
                config,
                verification_path=verification_path,
                rollout_path=rollout_path,
                output_dir=root / "labelled",
                labels_path=labels_path,
                samples_per_stratum=1,
            )
            report = read_json(report_path)
            internal = report["internal_verifier"]
            self.assertEqual(report["status"], "complete")
            self.assertEqual(internal["reviewed_samples"], 4)
            self.assertEqual(internal["confusion_matrix"]["unknown"]["pass"], 1)
            self.assertEqual(report["disagreements"]["count"], 1)
            self.assertTrue(
                report["official_benchmark"]["internal_verifier_not_used_as_primary_benchmark"]
            )
            self.assertTrue((root / "labelled/disagreements.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
