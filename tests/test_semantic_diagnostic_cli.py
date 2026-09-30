import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SemanticDiagnosticCliTests(unittest.TestCase):
    def test_windows_crlf_checkout_is_verified_against_original_run_receipt(self):
        row = {
            "id": "ep00-failure-f1",
            "reference": {"episode_index": "00", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"},
            "output": '{"phase":"Transport","state":"failure","failure_mode":"slip"}',
            "split_receipt_sha256": "frozen",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            line = json.dumps(row) + "\n"
            prediction = root / "predictions.jsonl"
            prediction.write_bytes(line.replace("\n", "\r\n").encode("utf-8"))
            original_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
            (root / "run_receipt.json").write_text(json.dumps({"output_sha256": {"predictions.jsonl": original_hash}}), encoding="utf-8")
            output = root / "diagnostic.json"
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "diagnose_semantic_predictions.py"),
                                        "--run", "base", str(prediction), "--output", str(output)],
                                       cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            evidence = json.loads(output.read_text(encoding="utf-8"))["runs"]["base"]
            self.assertIn("source_receipt_verification", evidence)
            self.assertEqual("CRLF_CHECKOUT_MATCH", evidence["source_receipt_verification"])
            self.assertEqual(original_hash, evidence["source_receipt_sha256"])

    def test_fenced_prediction_is_diagnostic_only_and_raw_file_is_unchanged(self):
        row = {
            "id": "ep00-failure-f1",
            "reference": {"episode_index": "00", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"},
            "output": '```json\n{"phase":"Transport","state":"failure","failure_mode":"slip"}\n```',
            "split_receipt_sha256": "fixed-split",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "predictions.jsonl"
            original = json.dumps(row) + "\n"
            prediction.write_text(original, encoding="utf-8")
            output = root / "diagnostic.json"
            command = [sys.executable, str(ROOT / "scripts" / "diagnose_semantic_predictions.py"),
                       "--run", "base", str(prediction), "--output", str(output)]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(0.0, result["runs"]["base"]["protocol"]["failure_recall"])
            self.assertEqual(1.0, result["runs"]["base"]["semantic"]["failure_recall"])
            self.assertEqual(1, result["runs"]["base"]["outer_fence_removed_count"])
            self.assertEqual(original, prediction.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
