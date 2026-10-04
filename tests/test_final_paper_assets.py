import unittest
from pathlib import Path

from scripts.generate_final_paper_assets import (
    assert_frozen_final_manifest, assert_same_final_references, load_final_evidence, render_final_assets,
)


class FinalPaperAssetTests(unittest.TestCase):
    def test_jointly_substituted_final_rows_cannot_pass_frozen_manifest(self):
        import json
        root = Path(__file__).resolve().parents[1]
        path = root / "artifacts/v2/runs/final-base-state/predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert_frozen_final_manifest(root, rows)
        bad = [{**row, "reference": {**row["reference"], "anchor_frame": row["reference"]["anchor_frame"] + 1}}
               for row in rows]
        with self.assertRaises(ValueError):
            assert_frozen_final_manifest(root, bad)

    def test_cross_schema_final_reference_mismatch_is_rejected(self):
        left = [{"id": "a", "reference": {"episode_index": "01", "execution_state": "failure"},
                 "split_receipt_sha256": "frozen"}]
        right = [{**left[0], "reference": {"episode_index": "01", "execution_state": "nominal"}}]
        with self.assertRaises(ValueError):
            assert_same_final_references({"State Base": left, "Full SFT": right})

    def test_saved_predictions_reproduce_frozen_primary_counts(self):
        root = Path(__file__).resolve().parents[1]
        result = load_final_evidence(root, bootstrap_samples=100)
        rows = result["runs"]
        self.assertEqual((rows["State Base"]["strict"]["json_valid_rate"], rows["State Base"]["semantic"]["failure_correct"]), (0.0, 18))
        self.assertEqual(rows["State Base"]["semantic"]["predicted_state_counts"]["failure"], 53)
        self.assertEqual([rows[f"State SFT {seed}"]["semantic"]["failure_correct"] for seed in (42, 43, 44)], [1, 7, 7])
        self.assertEqual(rows["Full SFT"]["semantic"]["failure_correct"], 0)
        self.assertEqual(result["contrasts"]["State SFT 43"]["second_minus_first"]["nominal_false_alarm_rate"]["point"], -10 / 18)
        self.assertIn("src/reboot_recovery/metrics.py", result["analysis_code_sha256"])

    def test_final_assets_come_from_saved_prediction_metrics(self):
        root = Path(__file__).resolve().parents[1]
        assets = render_final_assets(load_final_evidence(root, bootstrap_samples=100))
        table = assets["tables/final.tex"]
        self.assertIn("State Base & 0/54 & 18/18 & 18/18", table)
        self.assertIn("State SFT 43 & 54/54 & 7/18 & 8/18", table)
        self.assertIn("figures/final_tradeoff.tex", assets)
        self.assertIn("tables/paired_contrasts.tex", assets)

    def test_three_metric_axes_remain_distinguishable_without_color(self):
        root = Path(__file__).resolve().parents[1]
        figure = render_final_assets(load_final_evidence(root, bootstrap_samples=100))["figures/final_tradeoff.tex"]
        self.assertIn("fill=white", figure)
        self.assertIn("fill=black!65", figure)
        self.assertIn("pattern=north east lines", figure)
        self.assertIn("JSON", figure)
        self.assertIn("F-rec", figure)
        self.assertIn("N-FA", figure)


if __name__ == "__main__":
    unittest.main()
