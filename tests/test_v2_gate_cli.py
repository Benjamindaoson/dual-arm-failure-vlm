import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V2GateCliTests(unittest.TestCase):
    def test_matching_validation_predictions_open_gate_only_with_failure_gain(self):
        references = [
            {"episode_index": "00", "split": "val", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"},
            {"episode_index": "01", "split": "val", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "misalignment"},
            {"episode_index": "02", "split": "val", "phase_name": "Transport", "execution_state": "nominal", "failure_mode": "none"},
        ]
        def rows(states):
            return [{"id": f"sample-{index}", "reference": ref, "split_receipt_sha256": "frozen",
                     "output": json.dumps({"phase": "Transport", "state": state,
                                           "failure_mode": "none" if state == "nominal" else ref["failure_mode"]})}
                    for index, (ref, state) in enumerate(zip(references, states, strict=True))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, sft, output = root / "base.jsonl", root / "sft.jsonl", root / "gate.json"
            base.write_text("".join(json.dumps(row) + "\n" for row in rows(["recovery", "recovery", "nominal"])), encoding="utf-8")
            sft.write_text("".join(json.dumps(row) + "\n" for row in rows(["failure", "recovery", "nominal"])), encoding="utf-8")
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "decide_v2_rl_gate.py"),
                                        "--base-predictions", str(base), "--sft-predictions", str(sft),
                                        "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            decision = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("RUN_RLVR", decision["decision"])
            self.assertEqual("V2", decision["protocol"])
            self.assertTrue(decision["verifier_validated"])
            sft.write_text("".join(json.dumps(row) + "\n" for row in rows(["recovery", "recovery", "nominal"])), encoding="utf-8")
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "decide_v2_rl_gate.py"),
                                        "--base-predictions", str(base), "--sft-predictions", str(sft),
                                        "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertEqual("REVISIT_REPRESENTATION_OR_SUPERVISION", json.loads(output.read_text(encoding="utf-8"))["decision"])


if __name__ == "__main__":
    unittest.main()
