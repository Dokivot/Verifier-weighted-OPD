from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.config import load_config
from opd.training.audit import audit_sparse_kl


class SparseKLAuditTest(unittest.TestCase):
    def test_audit_is_deterministic_and_tracks_gradient_direction(self) -> None:
        config = load_config("configs/smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = audit_sparse_kl(config, root / "first.json")
            second = audit_sparse_kl(config, root / "second.json")
            self.assertEqual(first["mean_gradient_cosine"], second["mean_gradient_cosine"])
            self.assertGreater(first["mean_gradient_cosine"], 0.5)
            self.assertEqual(first["sparse_step_improvement_rate"], 1.0)
            self.assertTrue((root / "first.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
