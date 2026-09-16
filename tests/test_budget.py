from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.exceptions import OPDError
from opd.monitoring.budget import check_budget
from opd.tableio import write_json


class BudgetTest(unittest.TestCase):
    def test_budget_sums_jobs_and_enforces_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                root / "one/job_metrics.json",
                {"job_type": "rollout", "estimated_gpu_hours": 3},
            )
            config = {
                "paths": {"artifact_dir": str(root)},
                "project": {"gpu_hour_target": 5, "gpu_hour_hard_cap": 10},
            }
            self.assertEqual(check_budget(config)["gpu_hours"], 3.0)
            write_json(
                root / "two/job_metrics.json",
                {"job_type": "train", "estimated_gpu_hours": 8},
            )
            with self.assertRaises(OPDError):
                check_budget(config)


if __name__ == "__main__":
    unittest.main()
