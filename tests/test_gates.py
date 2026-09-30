from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery import gates

decide_stage = gates.decide_stage


class GateTests(unittest.TestCase):
    def test_v2_rejects_state_f1_gain_when_failure_signal_disappears(self):
        decide = getattr(gates, "decide_v2_rl_gate", None)
        self.assertIsNotNone(decide)
        base = {"task_schema": "full", "failure_correct": 6, "failure_support": 18, "state_macro_f1": 0.30}
        sft = {"task_schema": "full", "failure_correct": 0, "failure_support": 18, "state_macro_f1": 0.40, "strict_json_valid_rate": 1.0}
        self.assertEqual("REVISIT_REPRESENTATION_OR_SUPERVISION", decide(base, sft, verifier_validated=True)["decision"])

    def test_v2_requires_one_more_failure_hit_and_verified_reward(self):
        decide = getattr(gates, "decide_v2_rl_gate", None)
        self.assertIsNotNone(decide)
        base = {"task_schema": "full", "failure_correct": 6, "failure_support": 18, "state_macro_f1": 0.30}
        sft = {"task_schema": "full", "failure_correct": 7, "failure_support": 18, "state_macro_f1": 0.40, "strict_json_valid_rate": 1.0}
        self.assertEqual("VERIFY_REWARD_FIRST", decide(base, sft, verifier_validated=False)["decision"])
        self.assertEqual("RUN_RLVR", decide(base, sft, verifier_validated=True)["decision"])
        self.assertEqual(
            "REVISIT_REPRESENTATION_OR_SUPERVISION",
            decide(base, {**sft, "failure_correct": 6}, verifier_validated=True)["decision"],
        )
        self.assertEqual(
            "NO_OUTCOME_HEADROOM",
            decide(base, {**sft, "failure_correct": 18}, verifier_validated=True)["decision"],
        )
        self.assertEqual(
            "REVISIT_REPRESENTATION_OR_SUPERVISION",
            decide({**base, "pre_failure_false_positive_rate": 0.0},
                   {**sft, "pre_failure_false_positive_rate": 0.5}, verifier_validated=True)["decision"],
        )
        self.assertEqual(
            "REVISIT_REPRESENTATION_OR_SUPERVISION",
            decide({**base, "pre_failure_false_positive_rate": 0.0},
                   {**sft, "failure_correct": 18, "pre_failure_false_positive_rate": 0.5}, verifier_validated=True)["decision"],
        )
        with self.assertRaisesRegex(ValueError, "full-schema"):
            decide({**base, "task_schema": "state-only"}, sft, verifier_validated=True)

    def test_base_sufficient_stops_training(self):
        self.assertEqual("BASE_SUFFICIENT", decide_stage({"failure_recall": 0.91, "state_macro_f1": 0.86})["decision"])

    def test_sft_without_gain_revisits_task(self):
        decision = decide_stage(
            {"failure_recall": 0.5, "state_macro_f1": 0.5},
            {"failure_recall": 0.51, "state_macro_f1": 0.51},
        )
        self.assertEqual("REVISIT_DATA_OR_TASK", decision["decision"])

    def test_sft_gain_with_headroom_runs_rlvr(self):
        decision = decide_stage(
            {"failure_recall": 0.5, "state_macro_f1": 0.5},
            {"failure_recall": 0.7, "state_macro_f1": 0.7},
        )
        self.assertEqual("RUN_RLVR", decision["decision"])


if __name__ == "__main__":
    unittest.main()
