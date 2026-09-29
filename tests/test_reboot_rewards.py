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
        self.assertEqual(1.0, score_prediction(ref, out).total)

if __name__ == "__main__":
    unittest.main()
