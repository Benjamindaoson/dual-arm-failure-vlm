import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.annotations import build_training_windows, load_annotations, validate_episode

FIXTURE = {
    "dataset": "REBOOT26/sample_recovery-demonstrations",
    "task_id": "16mm-cylinder-install",
    "task_description": "Pick the 16mm cylinder and install on the taskboard",
    "fps": 30,
    "num_episodes": 2,
    "phase_names": ["Align (pick)", "Engage (pick)", "Transport", "Align (place)", "Engage (place)"],
    "episodes": [
        {
            "episode_index": "00", "task_id": "16mm-cylinder-install",
            "task_description": "Pick the 16mm cylinder and install on the taskboard",
            "episode_type": "recovery", "duration_frames": 897, "fps": 30,
            "phase_boundaries": {"tau_0": 0, "tau_1": 180, "tau_2": 570, "tau_3": 660, "tau_4": 780, "tau_5": 897},
            "phase_names": ["Align (pick)", "Engage (pick)", "Transport", "Align (place)", "Engage (place)"],
            "failure": {"originating_phase": 1, "failure_mode": "misalignment", "induced_at_frame": 180, "recovery_started_at_frame": 330, "language_description": "Realign and continue."}
        },
        {
            "episode_index": "01", "task_id": "16mm-cylinder-install",
            "task_description": "Pick the 16mm cylinder and install on the taskboard",
            "episode_type": "recovery", "duration_frames": 897, "fps": 30,
            "phase_boundaries": {"tau_0": 0, "tau_1": 180, "tau_2": 420, "tau_3": 480, "tau_4": 660, "tau_5": 897},
            "phase_names": ["Align (pick)", "Engage (pick)", "Transport", "Align (place)", "Engage (place)"],
            "failure": {"originating_phase": 2, "failure_mode": "slip", "induced_at_frame": 180, "recovery_started_at_frame": 330, "language_description": "Regrasp and continue."}
        }
    ]
}

class RebootAnnotationTests(unittest.TestCase):
    def _load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phase.json"
            p.write_text(json.dumps(FIXTURE), encoding="utf-8")
            return load_annotations(p)

    def test_phase_and_state_are_grounded_in_annotations(self):
        anns = self._load()
        ep = anns.episodes[0]
        self.assertEqual("Align (pick)", ep.phase_names[ep.phase_at(120)])
        self.assertEqual("nominal", ep.state_at(120))
        self.assertEqual("failure", ep.state_at(200))
        self.assertEqual("recovery", ep.state_at(400))
        self.assertEqual([], validate_episode(ep))

    def test_manifest_uses_episode_disjoint_splits(self):
        anns = self._load()
        windows, _ = build_training_windows(anns, n_frames=6)
        by_episode = {}
        for row in windows:
            by_episode.setdefault(row.episode_index, set()).add(row.split)
        self.assertTrue(all(len(v) == 1 for v in by_episode.values()))
        self.assertEqual(6, len(windows))
        self.assertEqual({"nominal", "failure", "recovery"}, {x.execution_state for x in windows})

    def test_bad_annotation_is_quarantined(self):
        fixture = json.loads(json.dumps(FIXTURE))
        fixture["episodes"][0]["failure"]["recovery_started_at_frame"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phase.json"
            p.write_text(json.dumps(fixture), encoding="utf-8")
            anns = load_annotations(p)
            windows, audit = build_training_windows(anns)
        self.assertIn("recovery_before_failure", audit["00"])
        self.assertTrue(all(x.episode_index != "00" for x in windows))

if __name__ == "__main__":
    unittest.main()
