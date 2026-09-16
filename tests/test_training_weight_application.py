from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from opd.config import load_config
from opd.schemas import TrainingRecord, VerificationStatus
from opd.tableio import read_json, write_records
from opd.training.mock import train_mock


class TrainingWeightApplicationTest(unittest.TestCase):
    def test_verifier_weight_scales_sample_loss(self) -> None:
        base_record = TrainingRecord(
            rollout_id="r1",
            sample_id="s1",
            prompt="Compute 1 + 1.",
            response="Final answer: 2",
            tokenizer_fingerprint="mock",
            input_ids=[10, 20],
            response_start=1,
            response_token_ids=[20],
            topk_token_ids=[[20, 21]],
            topk_logprobs=[[-0.1, -3.0]],
            tail_mass=[0.045],
            confidence_weights=[1.0],
            verifier_status=VerificationStatus.UNKNOWN,
            verifier_weight=1.0,
            method="verifier_opd",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "view.jsonl"
            config = deepcopy(load_config("configs/smoke.yaml"))
            config["training"]["input_path"] = str(input_path)

            write_records(input_path, [base_record.model_dump(mode="json")])
            config["training"]["output_dir"] = str(root / "full")
            full_path = train_mock(config)
            full_loss = float(read_json(full_path)["initial_loss"])

            reduced = base_record.model_copy(update={"verifier_weight": 0.3})
            write_records(input_path, [reduced.model_dump(mode="json")])
            config["training"]["output_dir"] = str(root / "reduced")
            reduced_path = train_mock(config)
            reduced_loss = float(read_json(reduced_path)["initial_loss"])

            self.assertAlmostEqual(reduced_loss, full_loss * 0.3, places=12)


if __name__ == "__main__":
    unittest.main()
