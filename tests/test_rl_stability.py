import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_rl_stability import check_run


class RLStabilityTests(unittest.TestCase):
    def test_rejects_nonfinite_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "run_receipt.json").write_text(json.dumps({"status": "COMPLETED"}))
            (path / "metrics.json").write_text(json.dumps({"global_step": 1}))
            (path / "train_log.json").write_text(json.dumps({"log_history": [
                {"loss": 0.1, "grad_norm": float("nan"), "reward": 0.5, "kl": 0.01}
            ]}))
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                check_run(path, 1)


if __name__ == "__main__":
    unittest.main()
