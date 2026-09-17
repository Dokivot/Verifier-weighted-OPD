from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from typing import Any


class BlackwellPackagingTest(unittest.TestCase):
    def test_linux_lock_uses_cuda_128_pytorch_stack(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)
        with (root / "uv.lock").open("rb") as handle:
            lock = tomllib.load(handle)

        index = project["tool"]["uv"]["index"]
        self.assertIn(
            {
                "name": "pytorch-cu128",
                "url": "https://download.pytorch.org/whl/cu128",
                "explicit": True,
            },
            index,
        )

        packages: list[dict[str, Any]] = lock["package"]
        expected = {
            "torch": "2.7.1+cu128",
            "torchaudio": "2.7.1+cu128",
            "torchvision": "0.22.1+cu128",
        }
        for name, version in expected.items():
            package = next(
                item for item in packages if item["name"] == name and item["version"] == version
            )
            self.assertEqual(
                package["source"]["registry"],
                "https://download.pytorch.org/whl/cu128",
            )

        cuda_runtime = next(item for item in packages if item["name"] == "nvidia-cuda-runtime-cu12")
        self.assertTrue(cuda_runtime["version"].startswith("12.8."))


if __name__ == "__main__":
    unittest.main()
