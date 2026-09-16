from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.monitoring.job import JobTimer
from opd.tableio import read_json


class JobMetricsTest(unittest.TestCase):
    def test_resume_preserves_prior_usage_for_same_artifact_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job_metrics.json"
            with JobTimer("rollout", path, artifact_run_id="same") as timer:
                timer.add(records=2, response_tokens=10)
            with JobTimer("rollout", path, artifact_run_id="same"):
                pass
            metrics = read_json(path)
            self.assertEqual(metrics["records"], 2)
            self.assertEqual(metrics["response_tokens"], 10)
            self.assertEqual(metrics["metadata"]["invocations"], 2)

            with JobTimer("rollout", path, artifact_run_id="different") as timer:
                timer.add(records=1, response_tokens=3)
            reset_metrics = read_json(path)
            self.assertEqual(reset_metrics["records"], 1)
            self.assertEqual(reset_metrics["response_tokens"], 3)
            self.assertEqual(reset_metrics["metadata"]["invocations"], 1)


if __name__ == "__main__":
    unittest.main()
