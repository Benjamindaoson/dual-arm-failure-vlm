import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_v2_method import choose


class MethodSelectionTests(unittest.TestCase):
    def test_failure_recall_precedes_average_f1(self):
        metrics = {
            "sparse-visual": {"failure_recall": 0.2, "state_macro_f1": 0.9},
            "dense-visual": {"failure_recall": 0.4, "state_macro_f1": 0.4},
            "sparse-trace": {"failure_recall": 0.1, "state_macro_f1": 0.95},
        }
        self.assertEqual("dense-visual", choose(metrics))

    def test_tie_prefers_simpler_input(self):
        metrics = {name: {"failure_recall": 0, "state_macro_f1": 0.5}
                   for name in ("sparse-visual", "dense-visual", "sparse-trace")}
        self.assertEqual("sparse-visual", choose(metrics))


if __name__ == "__main__":
    unittest.main()
