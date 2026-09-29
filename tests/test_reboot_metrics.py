from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.metrics import evaluate_predictions


class MetricsTests(unittest.TestCase):
    def test_failure_recall_and_invalid_json(self):
        rows = [
            {
                "reference": {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip", "anchor_kind": "failure"},
                "output": '{"phase":"Transport","state":"failure","failure_mode":"slip"}',
            },
            {
                "reference": {"phase_name": "Align (place)", "execution_state": "failure", "failure_mode": "misalignment", "anchor_kind": "failure"},
                "output": '{"phase":"Align (place)","state":"recovery","failure_mode":"misalignment"}',
            },
            {
                "reference": {"phase_name": "Align (pick)", "execution_state": "nominal", "failure_mode": "none", "anchor_kind": "nominal"},
                "output": "not-json",
            },
        ]
        metrics = evaluate_predictions(rows)
        self.assertAlmostEqual(2 / 3, metrics["json_valid_rate"])
        self.assertAlmostEqual(0.5, metrics["failure_recall"])
        self.assertEqual(3, metrics["n"])


if __name__ == "__main__":
    unittest.main()
