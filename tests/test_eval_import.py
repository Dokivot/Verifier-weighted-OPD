from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from opd.config import load_config
from opd.data.eval_import import fetch_evaluation_data, import_evaluation_data
from opd.tableio import read_json, read_records, write_records


class EvaluationImportTest(unittest.TestCase):
    def test_imports_generic_benchmark_export_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "math500_export.jsonl"
            write_records(
                source,
                [
                    {
                        "problem": "Compute 2 + 3.",
                        "answer": "5",
                        "solution": "2 + 3 = 5",
                        "subject": "algebra",
                    }
                ],
            )
            config = deepcopy(load_config("configs/smoke.yaml"))
            config["paths"]["data_dir"] = str(root / "data")
            output = import_evaluation_data(config, name="math500", input_path=source)
            records = read_records(output)
            self.assertEqual(records[0]["reference_answer"], "5")
            self.assertEqual(records[0]["split"], "evaluation:math500")
            manifest = read_json(output.with_suffix(".manifest.json"))
            self.assertEqual(manifest["metadata"]["name"], "math500")
            self.assertIn(str(source), manifest["files"])

    def test_fetches_pinned_huggingface_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(load_config("configs/smoke.yaml"))
            config["paths"]["data_dir"] = str(root / "data")
            config["evaluation_data"] = {
                "math500": {
                    "dataset_name": "HuggingFaceH4/MATH-500",
                    "dataset_subset": None,
                    "split": "test",
                    "revision": "fixed-revision",
                }
            }
            datasets_module = ModuleType("datasets")

            def fake_load_dataset(
                name: str, subset: str | None, *, split: str, revision: str
            ) -> list[dict[str, str]]:
                self.assertEqual(name, "HuggingFaceH4/MATH-500")
                self.assertIsNone(subset)
                self.assertEqual(split, "test")
                self.assertEqual(revision, "fixed-revision")
                return [{"problem": "1+1?", "answer": "2", "solution": "1+1=2"}]

            datasets_module.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
            with patch.dict("sys.modules", {"datasets": datasets_module}):
                output = fetch_evaluation_data(config, name="math500")
            rows = read_records(output)
            self.assertEqual(rows[0]["reference_answer"], "2")
            manifest = read_json(output.with_suffix(".manifest.json"))
            self.assertEqual(manifest["metadata"]["revision"], "fixed-revision")


if __name__ == "__main__":
    unittest.main()
