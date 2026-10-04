import json
import unittest
from pathlib import Path

from scripts import audit_submission


ROOT = Path(__file__).resolve().parents[1]


class WebpSchemaContrastTests(unittest.TestCase):
    def test_fixed_checkpoint_validation_contrast_is_bound_to_matching_inputs(self):
        verify = getattr(audit_submission, "verify_schema_contrast", None)
        self.assertIsNotNone(verify)
        result = verify(ROOT)
        self.assertEqual(result["sample_count"], 43)
        self.assertEqual(result["failure_support"], 15)
        self.assertEqual(result["state_only_semantic_failure_correct"], 15)
        self.assertEqual(result["full_schema_semantic_failure_correct"], 0)
        self.assertEqual(result["config_differences"], ["task_schema"])
        self.assertIn("artifacts/v2/runs/c2-base-val/predictions.jsonl", result["source_sha256"])

    def test_changed_reference_or_input_config_is_not_a_controlled_schema_contrast(self):
        check = getattr(audit_submission, "assert_matched_schema_inputs", None)
        self.assertIsNotNone(check)
        base = json.loads((ROOT / "artifacts/v2/runs/c2-base-val/config.json").read_text())
        full = json.loads((ROOT / "artifacts/v2/runs/multitask-base-val/config.json").read_text())
        rows = [json.loads(line) for line in
                (ROOT / "artifacts/v2/runs/c2-base-val/predictions.jsonl").read_text().splitlines()]
        other = [json.loads(line) for line in
                 (ROOT / "artifacts/v2/runs/multitask-base-val/predictions.jsonl").read_text().splitlines()]
        check(base, full, rows, other)
        with self.assertRaisesRegex(ValueError, "configuration"):
            check(base, {**full, "num_frames": 3}, rows, other)
        altered = [{**other[0], "reference": {**other[0]["reference"], "execution_state": "failure"}},
                   *other[1:]]
        with self.assertRaisesRegex(ValueError, "reference"):
            check(base, full, rows, altered)


if __name__ == "__main__":
    unittest.main()
