import unittest

from scripts.paired_compare import compare


class PairedCompareTests(unittest.TestCase):
    def test_nominal_false_alarm_is_paired_episode_bootstrapped(self):
        first = []
        second = []
        for episode in ("01", "02"):
            for state in ("nominal", "failure"):
                row = {"id": f"{episode}-{state}", "reference": {"episode_index": episode, "execution_state": state},
                       "split_receipt_sha256": "frozen", "output": '{"state":"failure"}'}
                first.append(row)
                second.append({**row, "output": '{"state":"nominal"}'})
        result = compare(first, second, task_schema="state-only", samples=50)
        self.assertEqual({"point": -1.0, "ci95": [-1.0, -1.0]},
                         result["second_minus_first"]["nominal_false_alarm_rate"])

    def test_episode_resampling_preserves_pairs(self):
        reference = {"episode_index": "01", "execution_state": "failure"}
        first = [{"id": "a", "reference": reference, "split_receipt_sha256": "split", "output": '{"state":"nominal"}'}]
        second = [{**first[0], "output": '{"state":"failure"}'}]
        result = compare(first, second, task_schema="state-only", samples=20)
        self.assertEqual(1, result["episode_support"])
        self.assertEqual({"point": 1.0, "ci95": [1.0, 1.0]}, result["second_minus_first"]["failure_recall"])
        self.assertEqual(1, result["paired_state_outcomes"]["failure"]["rescued"])
        with self.assertRaisesRegex(ValueError, "split receipt"):
            compare(first, [{**second[0], "split_receipt_sha256": "different"}], task_schema="state-only", samples=20)


if __name__ == "__main__":
    unittest.main()
