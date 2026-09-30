from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.ablations import resolve_ablation, select_window_frames, summarize_trace


class AblationTests(unittest.TestCase):
    def test_a0_uses_anchor_and_a3_uses_eight_images(self):
        row = {"anchor_frame": 50, "sampled_frames": [20, 40, 60, 80]}
        self.assertEqual([50], select_window_frames(row, resolve_ablation("A0")))
        a3 = resolve_ablation("A3")
        self.assertEqual(8, a3.num_frames * len(a3.cameras))
        self.assertTrue(a3.include_trace)

    def test_trace_text_records_dimensions_and_delta(self):
        summary = summarize_trace([[0, 1], [2, 4]], [[1, 2], [3, 5]])
        self.assertIn("state_delta=[2.0000,3.0000]", summary)
        self.assertIn("latest_action=[3.0000,5.0000]", summary)


if __name__ == "__main__":
    unittest.main()
