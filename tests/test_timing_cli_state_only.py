import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_failure_timing import main


class StateOnlyTimingCliTests(unittest.TestCase):
    def test_semantic_state_only_detection_and_false_alarm_rate(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            predictions = path / "predictions.jsonl"
            rows = [
                {"reference": {"episode_index": "07", "event_kind": "failure_onset", "relative_seconds": -0.5, "execution_state": "nominal"}, "output": '{"state":"nominal"}'},
                {"reference": {"episode_index": "07", "event_kind": "failure_onset", "relative_seconds": 0.0, "execution_state": "failure"}, "output": '```json\n{"state":"failure"}\n```'},
                {"reference": {"episode_index": "07", "event_kind": "failure_onset", "relative_seconds": 0.5, "execution_state": "failure"}, "output": '{"state":"failure"}'},
            ]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = path / "timing.json"
            self.assertEqual(0, main(["--predictions", str(predictions), "--task-schema", "state-only", "--semantic-diagnostic", "--output-json", str(output), "--output-csv", str(path / "timing.csv")]))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(0.5, result["episodes"]["07"]["detection_delay_seconds"])
            self.assertEqual(0.0, result["pre_onset_false_alarm_rate"])


if __name__ == "__main__":
    unittest.main()
