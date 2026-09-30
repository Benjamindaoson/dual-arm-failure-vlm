import unittest

from scripts.compare_task_schema_state import project_state
from scripts.paired_compare import compare


class CrossSchemaStateTests(unittest.TestCase):
    def test_full_schema_failure_miss_is_visible_after_projection(self):
        reference = {"episode_index": "01", "execution_state": "failure"}
        state = [{"id": "a", "reference": reference, "split_receipt_sha256": "split", "output": '{"state":"failure"}'}]
        full = [{**state[0], "output": '{"phase":"Transport","state":"recovery","failure_mode":"slip"}'}]
        result = compare(project_state(state, "state-only"), project_state(full, "full"),
                         task_schema="state-only", semantic=False, samples=20)
        self.assertEqual(1, result["paired_state_outcomes"]["failure"]["regressed"])


if __name__ == "__main__":
    unittest.main()
