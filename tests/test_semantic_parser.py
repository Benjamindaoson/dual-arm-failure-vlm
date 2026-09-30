from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reboot_recovery.metrics import predicted_state_distribution
from reboot_recovery.semantic_parser import normalize_semantic_output


class SemanticParserTests(unittest.TestCase):
    def test_only_whole_outer_fence_is_removed(self):
        body = '{"state":"failure"}'
        self.assertEqual(body, normalize_semantic_output(f"  ```json\n{body}\n```  "))
        self.assertEqual(body, normalize_semantic_output(f"```\n{body}\n```"))
        self.assertEqual(body, normalize_semantic_output(f"\n{body}\n"))
        self.assertEqual(f"Answer: ```\n{body}\n```", normalize_semantic_output(f"Answer: ```\n{body}\n```"))
        self.assertEqual(f"```JSON\n{body}\n```", normalize_semantic_output(f"```JSON\n{body}\n```"))
        self.assertEqual(f"```\n{body}\n``` extra", normalize_semantic_output(f"```\n{body}\n``` extra"))

    def test_collapse_metrics_use_only_valid_taxonomy(self):
        result = predicted_state_distribution(["nominal", "failure", "failure", "__invalid__"])
        self.assertEqual(2 / 3, result["collapse_ratio"])
        self.assertEqual(1, result["invalid_count"])
        self.assertAlmostEqual(0.6365141683, result["prediction_entropy_nats"])


if __name__ == "__main__":
    unittest.main()
