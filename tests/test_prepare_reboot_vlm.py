from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_reboot_vlm_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_reboot_vlm_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class PrepareDatasetTests(unittest.TestCase):
    def test_target_is_strict_structured_json(self):
        row = {"phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"}
        value = json.loads(module._target(row))
        self.assertEqual({"phase": "Transport", "state": "failure", "failure_mode": "slip"}, value)

    def test_jsonl_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.jsonl"
            p.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            self.assertEqual([{"a": 1}, {"a": 2}], module._read_jsonl(p))


if __name__ == "__main__":
    unittest.main()
