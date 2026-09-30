from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_v2_camera import select_camera


class CameraSelectionTests(unittest.TestCase):
    def test_validation_priority_and_fixed_tie_break(self):
        metrics = {
            "C0": {"failure_recall": 0, "state_macro_f1": 0.5},
            "C1": {"failure_recall": 0.2, "state_macro_f1": 0.4},
            "C2": {"failure_recall": 0.2, "state_macro_f1": 0.4},
        }
        self.assertEqual("C1", select_camera(metrics))
        metrics["C0"]["failure_recall"] = 0.3
        self.assertEqual("C0", select_camera(metrics))


if __name__ == "__main__":
    unittest.main()
