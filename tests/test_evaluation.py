import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upgraded_implementation" / "src"))

from multimodal_chart_gspo.evaluation import evaluate_records
from multimodal_chart_gspo.rewards import score_output


class ChartEvaluationTests(unittest.TestCase):
    def test_correct_but_malformed_is_scored_separately(self):
        score = score_output("42", "The result is 42")
        self.assertEqual(1.0, score.correctness)
        self.assertEqual(0.0, score.format_reward)

    def test_record_aggregation(self):
        report = evaluate_records([
            {"reference_answer": "42", "output": "<answer>42</answer>", "task": "chart"},
            {"reference_answer": "blue", "output": "<answer>red</answer>", "task": "color"},
        ])
        self.assertEqual(2, report["total"])
        self.assertEqual(0.5, report["accuracy"])
        self.assertEqual(1.0, report["format_rate"])


if __name__ == "__main__":
    unittest.main()
