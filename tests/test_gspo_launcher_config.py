import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_gspo import load_training_config


ROOT = Path(__file__).resolve().parents[1]


class GSPOLauncherConfigTests(unittest.TestCase):
    def test_v100_smoke_profile_is_fp16_and_small_enough(self):
        config = load_training_config(ROOT / "configs" / "gpu_gspo.json")
        self.assertEqual("fp16", config["precision"])
        self.assertLessEqual(config["model_parameter_billion"], 3)

    def test_v100_rejects_large_model(self):
        payload = json.loads((ROOT / "configs" / "gpu_gspo.json").read_text(encoding="utf-8"))
        payload["model_parameter_billion"] = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "single_v100"):
                load_training_config(path)
