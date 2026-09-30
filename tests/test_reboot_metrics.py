from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery import metrics

evaluate_predictions = metrics.evaluate_predictions


class MetricsTests(unittest.TestCase):
    def test_semantic_normalization_removes_only_an_outer_json_fence(self):
        normalizer = getattr(metrics, "normalize_semantic_output", None)
        self.assertIsNotNone(normalizer)
        body = '{"phase":"Transport","state":"failure","failure_mode":"slip"}'
        self.assertEqual(body, normalizer("  ```json\n" + body + "\n```  "))
        self.assertEqual(body, normalizer("\n" + body + "\n"))
        self.assertEqual("Answer: ```json\n" + body + "\n```", normalizer("Answer: ```json\n" + body + "\n```"))
        self.assertEqual("```JSON\n" + body + "\n```", normalizer("```JSON\n" + body + "\n```"))

    def test_diagnostic_keeps_strict_score_and_counts_recovered_failure(self):
        diagnostic = getattr(metrics, "diagnose_predictions", None)
        self.assertIsNotNone(diagnostic)
        rows = [
            {
                "reference": {"episode_index": "00", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"},
                "output": '```json\n{"phase":"Transport","state":"failure","failure_mode":"slip"}\n```',
            },
            {
                "reference": {"episode_index": "01", "phase_name": "Transport", "execution_state": "nominal", "failure_mode": "none"},
                "output": 'Answer: ```json\n{"phase":"Transport","state":"nominal","failure_mode":"none"}\n```',
            },
        ]
        original = [row["output"] for row in rows]
        result = diagnostic(rows, bootstrap_samples=0)
        self.assertEqual(0.0, result["protocol"]["json_valid_rate"])
        self.assertEqual(0.5, result["semantic"]["json_valid_rate"])
        self.assertEqual(1.0, result["semantic"]["failure_recall"])
        self.assertEqual(1, result["semantic_exact_count"])
        self.assertEqual({"failure": 1, "__invalid__": 1}, result["semantic_predicted_state_counts"])
        self.assertEqual(1, result["outer_fence_removed_count"])
        self.assertEqual(original, [row["output"] for row in rows])

    def test_failure_recall_and_invalid_json(self):
        rows = [
            {
                "reference": {"episode_index": "00", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip", "anchor_kind": "failure"},
                "output": '{"phase":"Transport","state":"failure","failure_mode":"slip"}',
            },
            {
                "reference": {"episode_index": "01", "phase_name": "Align (place)", "execution_state": "failure", "failure_mode": "misalignment", "anchor_kind": "failure"},
                "output": '{"phase":"Align (place)","state":"recovery","failure_mode":"misalignment"}',
            },
            {
                "reference": {"episode_index": "02", "phase_name": "Align (pick)", "execution_state": "nominal", "failure_mode": "none", "anchor_kind": "nominal"},
                "output": "not-json",
            },
        ]
        metrics = evaluate_predictions(rows)
        self.assertAlmostEqual(2 / 3, metrics["json_valid_rate"])
        self.assertAlmostEqual(0.5, metrics["failure_recall"])
        self.assertIn("failure_correct", metrics)
        self.assertIn("failure_support", metrics)
        self.assertEqual(1, metrics["failure_correct"])
        self.assertEqual(2, metrics["failure_support"])
        self.assertEqual(3, metrics["n"])
        self.assertEqual("episode", metrics["episode_bootstrap"]["unit"])
        self.assertIn("recovery_recall", metrics)


if __name__ == "__main__":
    unittest.main()
