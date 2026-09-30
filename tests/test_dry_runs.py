import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def prepared_root(root: Path) -> Path:
    dataset = root / "prepared"
    dataset.mkdir()
    images = []
    for index in range(8):
        path = dataset / f"image-{index}.jpg"
        path.write_bytes(b"placeholder")
        images.append(path.name)
    row = {
        "id": "ep00-failure", "images": images,
        "prompt": [{"role": "user", "content": [{"type": "image"}] * 8 + [{"type": "text", "text": "diagnose"}]}],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": "{}"}]}],
        "reference": {"episode_index": "00", "phase_name": "Transport", "execution_state": "failure", "failure_mode": "slip"},
    }
    for split in ("train", "val", "test"):
        (dataset / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return dataset


class DryRunTests(unittest.TestCase):
    def test_v2_rl_dry_run_requires_v2_gate_and_records_reward_scheme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = prepared_root(root)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"task_schema":"full"}', encoding="utf-8")
            gate = root / "gate.json"
            gate.write_text('{"decision":"RUN_RLVR","protocol":"V2","task_schema":"full","verifier_validated":true}', encoding="utf-8")
            output = root / "rl-v2"
            self.assertTrue((ROOT / "configs" / "reboot_grpo_v2.json").is_file())
            args = ["--config", str(ROOT / "configs" / "reboot_grpo_v2.json"),
                    "--gate-decision", str(gate), "--dataset-root", str(dataset),
                    "--sft-checkpoint", str(checkpoint), "--output-dir", str(output), "--dry-run"]
            self.assertEqual(0, load_script("train_rlvr").main(args))
            summary = json.loads((output / "dry_run.json").read_text(encoding="utf-8"))
            self.assertIn("reward_scheme", summary)
            self.assertEqual("state_gated_v2", summary["reward_scheme"])
            (checkpoint / "config.json").write_text('{"task_schema":"state-only"}', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "full-schema SFT checkpoint"):
                load_script("train_rlvr").main(args[:-3] + ["--output-dir", str(root / "wrong-checkpoint"), "--dry-run"])
            (checkpoint / "config.json").write_text('{"task_schema":"full"}', encoding="utf-8")
            gate.write_text('{"decision":"RUN_RLVR"}', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "V2 gate"):
                load_script("train_rlvr").main(args[:-3] + ["--output-dir", str(root / "rejected"), "--dry-run"])

    def test_base_sft_and_rlvr_dry_runs_leave_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = prepared_root(root)
            base_out = root / "base"
            self.assertEqual(0, load_script("eval_base").main([
                "--dataset-root", str(dataset), "--output-dir", str(base_out), "--dry-run"
            ]))
            self.assertEqual("DRY_RUN_VERIFIED", json.loads((base_out / "run_receipt.json").read_text())["status"])

            sft_out = root / "sft"
            self.assertEqual(0, load_script("train_sft").main([
                "--dataset-root", str(dataset), "--output-dir", str(sft_out), "--dry-run"
            ]))
            self.assertEqual("DRY_RUN_VERIFIED", json.loads((sft_out / "run_receipt.json").read_text())["status"])

            gate = root / "gate.json"
            gate.write_text('{"decision":"RUN_RLVR"}', encoding="utf-8")
            rl_out = root / "rl"
            self.assertEqual(0, load_script("train_rlvr").main([
                "--config", str(ROOT / "configs" / "reboot_grpo.json"),
                "--gate-decision", str(gate), "--dataset-root", str(dataset),
                "--sft-checkpoint", str(sft_out), "--output-dir", str(rl_out), "--dry-run",
            ]))
            receipt = json.loads((rl_out / "run_receipt.json").read_text())
            self.assertEqual("DRY_RUN_VERIFIED", receipt["status"])


if __name__ == "__main__":
    unittest.main()
