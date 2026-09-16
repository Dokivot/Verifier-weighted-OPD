from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from opd.hashing import stable_hash
from opd.tableio import atomic_write_text, read_jsonl, write_jsonl


class ArtifactTest(unittest.TestCase):
    def test_hash_is_order_independent_for_mappings(self) -> None:
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))

    def test_atomic_writes_leave_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            atomic_write_text(path, "complete")
            self.assertEqual(path.read_text(encoding="utf-8"), "complete")
            self.assertFalse(path.with_suffix(".txt.partial").exists())

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(path, [{"id": 1}, {"id": 2}])
            self.assertEqual(read_jsonl(path), [{"id": 1}, {"id": 2}])


if __name__ == "__main__":
    unittest.main()
