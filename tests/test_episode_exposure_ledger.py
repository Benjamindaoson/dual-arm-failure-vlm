import tempfile
import unittest
from pathlib import Path

from scripts.build_episode_exposure_ledger import build_ledger, validate_ledger


class EpisodeExposureLedgerTests(unittest.TestCase):
    def test_real_frozen_sources_cover_all_original_episodes(self):
        root = Path(__file__).resolve().parents[1]
        rows, sources = build_ledger(root)
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(row["usable"] == "yes" for row in rows), 53)
        self.assertEqual(sum(row["usable"] == "no" for row in rows), 7)
        self.assertEqual(sum(row["untouched_same_task"] == "yes" for row in rows), 0)
        tier_b = [row for row in rows if row["final_tier_b_test"] == "yes"]
        self.assertEqual([row["episode_id"] for row in tier_b], ["11", "23", "29", "33", "51", "58"])
        self.assertTrue(all(row["v1_split"] == "train" for row in tier_b))
        self.assertTrue(all(row["evidence_tier"] == "internal_exposed_test" for row in tier_b))
        self.assertIn("artifacts/splits/split_receipt.json", sources)

    def test_validator_detects_tampered_exposure(self):
        root = Path(__file__).resolve().parents[1]
        rows, sources = build_ledger(root)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            from scripts.build_episode_exposure_ledger import write_ledger
            write_ledger(path, rows)
            validate_ledger(root, path)
            body = path.read_text(encoding="utf-8")
            path.write_text(body.replace("internal_exposed_test", "independent_test", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_ledger(root, path)


if __name__ == "__main__":
    unittest.main()
