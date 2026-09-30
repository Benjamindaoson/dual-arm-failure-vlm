from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery import rewards

score_prediction = rewards.score_prediction

class RebootRewardTests(unittest.TestCase):
    def test_state_gated_reward_denies_mode_credit_when_state_is_wrong(self):
        scorer = getattr(rewards, "score_state_gated_prediction", None)
        self.assertIsNotNone(scorer)
        ref = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "misalignment"}
        wrong_state = '{"phase":"Transport","state":"recovery","failure_mode":"misalignment"}'
        right_state = '{"phase":"Transport","state":"failure","failure_mode":"misalignment"}'
        self.assertEqual(-1.0, scorer(ref, wrong_state))
        self.assertEqual(1.0, scorer(ref, right_state))
        self.assertEqual(0.0, scorer(ref, "not-json"))

    def test_state_gated_nominal_row_can_reach_full_reward_without_mode_shortcut(self):
        scorer = getattr(rewards, "score_state_gated_prediction", None)
        self.assertIsNotNone(scorer)
        ref = {"phase_name": "Align (pick)", "execution_state": "nominal", "failure_mode": "none"}
        output = '{"phase":"Align (pick)","state":"nominal","failure_mode":"none"}'
        self.assertEqual(1.0, scorer(ref, output))

    def test_state_gated_wrong_states_and_modes(self):
        scorer = rewards.score_state_gated_prediction
        refs = {
            "failure": {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"},
            "recovery": {"phase_name": "Transport", "execution_state": "recovery", "failure_mode": "slip"},
            "nominal": {"phase_name": "Transport", "execution_state": "nominal", "failure_mode": "none"},
        }
        def output(state, mode):
            import json
            return json.dumps({"phase": "Transport", "state": state, "failure_mode": mode})
        self.assertEqual(-1.0, scorer(refs["failure"], output("nominal", "none")))
        self.assertEqual(-1.0, scorer(refs["failure"], output("recovery", "slip")))
        self.assertEqual(0.0, scorer(refs["recovery"], output("failure", "slip")))
        self.assertEqual(0.0, scorer(refs["nominal"], output("failure", "slip")))
        self.assertAlmostEqual(0.8, scorer(refs["failure"], output("failure", "jamming")))
        self.assertEqual(1.0, scorer(refs["failure"], output("failure", "slip")))
        self.assertEqual(0.0, scorer(refs["failure"], "not json"))
        self.assertEqual(0.0, scorer(refs["failure"], output("failure", "unknown")))

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
