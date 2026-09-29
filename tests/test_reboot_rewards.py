from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.rewards import score_prediction

class RebootRewardTests(unittest.TestCase):
    def test_invalid_json_gets_zero(self):
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        self.assertEqual(0.0, score_prediction(ref, "Transport / failure / slip").total)

    def test_exact_structured_diagnosis_gets_full_reward(self):
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        out = '{"phase":"Transport","state":"failure","failure_mode":"slip"}'
        self.assertEqual(1.0, score_prediction(ref, out).total)

    def test_format_is_gate_not_reward_shortcut(self):
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        out = '{"phase":"Align (pick)","state":"recovery","failure_mode":"misalignment"}'
        score = score_prediction(ref, out)
        self.assertEqual(1.0, score.valid_json)
        self.assertEqual(0.0, score.total)

    def test_nominal_does_not_reward_failure_taxonomy_guessing(self):
        ref = {"phase_name": "Align (pick)", "execution_state": "nominal", "failure_mode": "none"}
        out = '{"phase":"Align (pick)","state":"nominal","failure_mode":"slip"}'
        self.assertEqual(0.0, score_prediction(ref, out).total)

    def test_unknown_state_label_gets_no_task_reward(self):
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        score = score_prediction(ref, '{"phase":"Transport","state":"unknown","failure_mode":"slip"}')
        self.assertEqual(0.0, score.valid_json)
        self.assertEqual(0.0, score.total)

    def test_unknown_phase_and_failure_labels_are_invalid(self):
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        unknown_phase = '{"phase":"bogus","state":"failure","failure_mode":"slip"}'
        unknown_mode = '{"phase":"Transport","state":"failure","failure_mode":"bogus"}'
        self.assertEqual(0.0, score_prediction(ref, unknown_phase).valid_json)
        self.assertEqual(0.0, score_prediction(ref, unknown_mode).valid_json)

    def test_extra_json_keys_are_invalid(self):
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        out = '{"phase":"Transport","state":"failure","failure_mode":"slip","rationale":"x"}'
        self.assertEqual(0.0, score_prediction(ref, out).valid_json)

if __name__ == "__main__":
    unittest.main()
