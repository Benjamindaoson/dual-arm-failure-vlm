import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PredictionComparisonTests(unittest.TestCase):
    def test_paired_outcomes_require_identical_receipts(self):
        spec = importlib.util.spec_from_file_location("compare_predictions", ROOT / "scripts" / "compare_model_predictions.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        reference = {"episode_index": "01", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            left = {"id": "x", "reference": reference, "output": "{}", "split_receipt_sha256": "abc"}
            right = {**left, "output": '{"phase":"Transport","state":"failure","failure_mode":"slip"}'}
            (base / "left.jsonl").write_text(json.dumps(left) + "\n", encoding="utf-8")
            (base / "right.jsonl").write_text(json.dumps(right) + "\n", encoding="utf-8")
            result = module.main([
                "--left", str(base / "left.jsonl"), "--right", str(base / "right.jsonl"),
                "--output-json", str(base / "out.json"), "--output-csv", str(base / "out.csv"),
            ])
            payload = json.loads((base / "out.json").read_text(encoding="utf-8"))
        self.assertEqual(0, result)
        self.assertEqual(1, payload["paired_outcomes"]["left_wrong_right_correct"])


if __name__ == "__main__":
    unittest.main()
