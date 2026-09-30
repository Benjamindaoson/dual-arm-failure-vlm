from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_v2_final_split import V1_DIAGNOSTIC_SET, assign_fallback


class V2FinalSplitTests(unittest.TestCase):
    def test_fallback_is_deterministic_and_excludes_v1_diagnostic(self):
        usable = {f"{index:02d}" for index in range(60)}
        first = assign_fallback(usable, seed=2026)
        self.assertEqual(first, assign_fallback(set(reversed(sorted(usable))), seed=2026))
        self.assertFalse(V1_DIAGNOSTIC_SET & set(first))
        self.assertEqual(6, sum(split == "test" for split in first.values()))
        self.assertEqual(5, sum(split == "val" for split in first.values()))
        self.assertEqual(len(usable) - 6, len(first))


if __name__ == "__main__":
    unittest.main()
