from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.gates import decide_stage


class GateTests(unittest.TestCase):
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
