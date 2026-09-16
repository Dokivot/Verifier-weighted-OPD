from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.config import load_config
from opd.exceptions import ConfigurationError


class ConfigTest(unittest.TestCase):
    def test_single_seed_policy_accepts_registered_seed(self) -> None:
        config = load_config("configs/smoke.yaml")
        self.assertEqual(config["project"]["seed"], 42)
        self.assertEqual(config["project"]["registered_seeds"], [42])

    def test_single_seed_policy_rejects_multiple_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                "project:\n  seed: 42\n  seed_policy: single\n  registered_seeds: [42, 43]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
