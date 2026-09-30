from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery import metrics, prompts


class StateOnlyTests(unittest.TestCase):
    def test_training_row_keeps_images_and_reference_but_removes_other_targets(self):
        transform = getattr(prompts, "to_state_only_record", None)
        self.assertIsNotNone(transform)
        original = {
            "id": "ep02-failure-f200",
            "images": ["images/first.jpg", "images/second.jpg"],
            "trace_summary": "state_delta=[0.1]",
            "reference": {"task_description": "Install the cylinder", "execution_state": "failure", "phase_name": "Transport", "failure_mode": "slip"},
            "split_receipt_sha256": "frozen",
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "old"}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "old"}]}],
        }
        transformed = transform(original)
        self.assertEqual(original["images"], transformed["images"])
        self.assertEqual(original["reference"], transformed["reference"])
        self.assertEqual("frozen", transformed["split_receipt_sha256"])
        self.assertEqual('{"state":"failure"}', transformed["completion"][0]["content"][0]["text"])
        self.assertEqual(2, sum(block["type"] == "image" for block in transformed["prompt"][0]["content"]))
        prompt = transformed["prompt"][0]["content"][-1]["text"]
        self.assertIn("Install the cylinder", prompt)
        self.assertIn("state_delta=[0.1]", prompt)
        self.assertNotIn("failure_mode", prompt)
        self.assertEqual("old", original["prompt"][0]["content"][-1]["text"])

    def test_one_key_state_metrics_do_not_invent_phase_or_mode_scores(self):
        evaluate = getattr(metrics, "evaluate_state_predictions", None)
        self.assertIsNotNone(evaluate)
        rows = [
            {"reference": {"episode_index": "00", "execution_state": "failure"}, "output": '{"state":"failure"}'},
            {"reference": {"episode_index": "01", "execution_state": "failure"}, "output": '{"state":"failure","phase":"Transport"}'},
            {"reference": {"episode_index": "02", "execution_state": "nominal"}, "output": '{"state":"nominal"}'},
        ]
        result = evaluate(rows, bootstrap_samples=0)
        self.assertEqual(2 / 3, result["json_valid_rate"])
        self.assertEqual(1 / 2, result["failure_recall"])
        self.assertEqual({"failure": 1, "nominal": 1, "__invalid__": 1}, result["predicted_state_counts"])
        self.assertNotIn("phase_macro_f1", result)
        self.assertNotIn("failure_mode_macro_f1", result)

    def test_existing_train_and_eval_commands_accept_state_only_dry_run(self):
        row = {
            "id": "ep02-failure-f200", "images": ["image.jpg"],
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "old"}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "old"}]}],
            "reference": {"episode_index": "02", "task_description": "Install cylinder", "execution_state": "failure", "phase_name": "Transport", "failure_mode": "slip"},
            "split_receipt_sha256": "frozen",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").touch()
            for split in ("train", "val", "test"):
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            for script, extra in (
                ("train_sft.py", []),
                ("eval_base.py", ["--num-frames", "1", "--cameras", "observation.images.cam_high"]),
            ):
                output = root / script
                command = [sys.executable, str(ROOT / "scripts" / script), "--dataset-root", str(root),
                           "--task-schema", "state-only", "--output-dir", str(output), "--dry-run", *extra]
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(0, completed.returncode, f"{script}: {completed.stderr}")
                config = json.loads((output / "config.json").read_text(encoding="utf-8"))
                self.assertEqual("state-only", config["task_schema"])


if __name__ == "__main__":
    unittest.main()
