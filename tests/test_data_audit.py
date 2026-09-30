import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.audit import audit_dataset


class DataAuditTests(unittest.TestCase):
    def test_audit_quarantines_bad_rows_and_records_schema(self):
        info = {
            "codebase_version": "v3.0", "total_episodes": 2, "total_frames": 21, "fps": 30,
            "features": {
                "action": {"dtype": "float32", "shape": [14]},
                "observation.state": {"dtype": "float32", "shape": [14]},
                "observation.images.cam_high": {"dtype": "video", "shape": [2, 2, 3], "info": {"video.is_depth_map": False}},
                "episode_index": {"dtype": "int64", "shape": [1]},
            },
        }
        base = {
            "task_id": "task", "task_description": "task", "episode_type": "recovery",
            "duration_frames": 10, "fps": 30,
            "phase_names": ["p1", "p2"], "phase_boundaries": {"tau_0": 0, "tau_1": 5, "tau_2": 10},
            "failure": {"originating_phase": 1, "failure_mode": "slip", "induced_at_frame": 3,
                        "recovery_started_at_frame": 6, "language_description": "recover"},
        }
        good = {**base, "episode_index": "00"}
        bad = json.loads(json.dumps({**base, "episode_index": "01"}))
        bad["failure"]["originating_phase"] = 9
        bad["failure"]["recovery_started_at_frame"] = 2
        phase = {"dataset": "repo", "task_id": "task", "task_description": "task", "fps": 30,
                 "num_episodes": 2, "phase_names": ["p1", "p2"], "episodes": [good, bad]}
        result = audit_dataset(info, phase, {"id": "repo", "sha": "abc", "cardData": {"license": "apache-2.0"}})
        self.assertEqual([14], result.schema["robot_state"]["shape"])
        self.assertEqual(["observation.images.cam_high"], result.schema["rgb_camera_keys"])
        self.assertFalse(result.schema["has_depth"])
        self.assertEqual(["01"], result.annotation_audit["quarantined_episode_ids"])
        self.assertIn("originating_phase_out_of_range", result.annotation_audit["issues_by_episode"]["01"])
        self.assertIn("recovery_before_failure", result.annotation_audit["issues_by_episode"]["01"])
        self.assertIn("annotated_duration_sum_mismatch", result.annotation_audit["global_issues"])


if __name__ == "__main__":
    unittest.main()
