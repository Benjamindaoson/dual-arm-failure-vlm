import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.dataset import verify_local_dataset


class DatasetLocalModeTests(unittest.TestCase):
    def test_metadata_only_local_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text('{"total_episodes": 1}', encoding="utf-8")
            (root / "meta" / "phase.json").write_text('{"episodes": []}', encoding="utf-8")
            result = verify_local_dataset(root, require_full=False)
            self.assertTrue(result["valid"])
            self.assertFalse(result["full_payload_present"])

    def test_missing_metadata_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_local_dataset(Path(tmp), require_full=False)
        self.assertFalse(result["valid"])
        self.assertIn("meta/info.json", result["missing"])

    def test_empty_payload_directories_do_not_count_as_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            (root / "meta" / "phase.json").write_text("{}", encoding="utf-8")
            (root / "data").mkdir()
            (root / "videos").mkdir()
            result = verify_local_dataset(root, require_full=True)
        self.assertFalse(result["valid"])
        self.assertFalse(result["full_payload_present"])


if __name__ == "__main__":
    unittest.main()
