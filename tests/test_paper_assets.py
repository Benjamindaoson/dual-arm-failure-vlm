import unittest

from scripts.generate_paper_assets import render_v1_assets


class PaperAssetTests(unittest.TestCase):
    def test_table_and_figure_use_input_metrics(self):
        names = ("Base", "SFT", "GRPO", "GSPO")
        runs = {name: {"protocol": {"json_valid_rate": 0.25, "state_macro_f1": 0.2},
                       "semantic": {"state_macro_f1": 0.4, "failure_recall": 0.75}}
                for name in names}
        distribution = {name: {"collapse_ratio": 0.5,
                               "probabilities": {"nominal": 0.5, "failure": 0.25, "recovery": 0.25}}
                        for name in names}
        output = render_v1_assets({"runs": runs}, distribution)
        self.assertIn("Base & 25.0 & 20.0 & 40.0 & 75.0 & 50.0", output["tables/v1.tex"])
        self.assertIn("rectangle (0.86,0.7500)", output["figures/v1_progression.tex"])
        with self.assertRaises(ValueError):
            render_v1_assets({"runs": {"Base": runs["Base"]}}, distribution)


if __name__ == "__main__":
    unittest.main()
