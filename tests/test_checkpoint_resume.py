from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.checkpoints import resolve_checkpoint


class CheckpointResumeTests(unittest.TestCase):
    def test_latest_checkpoint_is_selected_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint-9").mkdir()
            (root / "checkpoint-10").mkdir()
            self.assertEqual(root / "checkpoint-10", resolve_checkpoint(root, "latest"))

    def test_missing_explicit_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(Path(tmp), "checkpoint-1")


if __name__ == "__main__":
    unittest.main()
