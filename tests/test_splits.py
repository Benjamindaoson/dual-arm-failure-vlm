from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.annotations import EpisodeAnnotation, FailureAnnotation, split_episode_ids
from reboot_recovery.splits import build_split_receipt


def episode(index: int) -> EpisodeAnnotation:
    return EpisodeAnnotation(
        episode_index=f"{index:02d}", task_id="task", task_description="task", episode_type="recovery",
        duration_frames=100, fps=10, phase_boundaries=(0, 20, 40, 60, 80, 100),
        phase_names=("p1", "p2", "p3", "p4", "p5"),
        failure=FailureAnnotation((index % 5) + 1, "slip", 30, 50, "recover"),
    )


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_episode_disjoint(self):
        episodes = [episode(i) for i in range(10)]
        first = split_episode_ids(episodes, seed=42)
        second = split_episode_ids(reversed(episodes), seed=42)
        self.assertEqual(first, second)
        self.assertEqual({"train": 8, "val": 1, "test": 1}, {s: list(first.values()).count(s) for s in set(first.values())})
        receipt = build_split_receipt(episodes, first, seed=42)
        self.assertEqual(64, len(receipt["manifest_sha256"]))
        self.assertEqual(10, len({e for ids in receipt["episodes"].values() for e in ids}))


if __name__ == "__main__":
    unittest.main()
