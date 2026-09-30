from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.timing import evaluate_failure_timing


class TimingMetricTests(unittest.TestCase):
    def test_delay_requires_consecutive_failure_predictions(self):
        rows = [
            {"episode_index": "00", "relative_seconds": -1.0, "predicted_state": "failure"},
            {"episode_index": "00", "relative_seconds": 0.0, "predicted_state": "nominal"},
            {"episode_index": "00", "relative_seconds": 0.5, "predicted_state": "failure"},
            {"episode_index": "00", "relative_seconds": 1.0, "predicted_state": "failure"},
        ]
        result = evaluate_failure_timing(rows, consecutive=2)
        self.assertEqual(1.0, result["episodes"]["00"]["detection_delay_seconds"])
        self.assertEqual(1, result["episodes"]["00"]["false_alarms_before_failure"])
        self.assertEqual(1.0, result["pre_onset_false_alarm_rate"])

    def test_undetected_episode_is_explicit(self):
        result = evaluate_failure_timing([
            {"episode_index": "00", "relative_seconds": 0.0, "predicted_state": "nominal"}
        ], consecutive=1)
        self.assertIsNone(result["episodes"]["00"]["detection_delay_seconds"])
        self.assertEqual(1, result["undetected_episodes"])
        self.assertIsNone(result["pre_onset_false_alarm_rate"])

    def test_failure_predictions_during_recovery_do_not_count_as_detection(self):
        rows = [
            {"episode_index": "00", "relative_seconds": 0.0, "gold_state": "failure", "predicted_state": "nominal"},
            {"episode_index": "00", "relative_seconds": 0.5, "gold_state": "recovery", "predicted_state": "failure"},
            {"episode_index": "00", "relative_seconds": 1.0, "gold_state": "recovery", "predicted_state": "failure"},
        ]
        result = evaluate_failure_timing(rows, consecutive=2)
        self.assertEqual(1, result["undetected_episodes"])

    def test_recovery_onset_uses_recovery_as_target_state(self):
        rows = [
            {"episode_index": "00", "relative_seconds": -0.5, "gold_state": "failure", "predicted_state": "recovery"},
            {"episode_index": "00", "relative_seconds": 0.0, "gold_state": "recovery", "predicted_state": "recovery"},
            {"episode_index": "00", "relative_seconds": 0.5, "gold_state": "recovery", "predicted_state": "recovery"},
        ]
        result = evaluate_failure_timing(rows, consecutive=2, target_state="recovery")
        self.assertEqual("recovery", result["target_state"])
        self.assertEqual(0.5, result["episodes"]["00"]["detection_delay_seconds"])
        self.assertEqual(1, result["episodes"]["00"]["false_alarms_before_onset"])


if __name__ == "__main__":
    unittest.main()
