from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.config import validate_rl_config


class ConfigValidationTests(unittest.TestCase):
    def test_v2_state_gated_reward_uses_one_grpo_reward_function(self):
        base = {"algorithm": "grpo", "importance_sampling_level": "token", "loss_type": "grpo",
                "num_generations": 4, "max_completion_length": 96,
                "reward_scheme": "state_gated_v2", "reward_weights": [1.0]}
        self.assertEqual("state_gated_v2", validate_rl_config(base)["reward_scheme"])
        with self.assertRaisesRegex(ValueError, "reward_weights"):
            validate_rl_config({**base, "reward_weights": [0.25, 0.35, 0.40]})
        with self.assertRaisesRegex(ValueError, "GRPO"):
            validate_rl_config({**base, "algorithm": "gspo", "importance_sampling_level": "sequence"})

    def test_grpo_and_gspo_semantics_are_distinct(self):
        grpo = validate_rl_config({"algorithm": "grpo", "importance_sampling_level": "token", "loss_type": "grpo", "num_generations": 4, "max_completion_length": 96})
        gspo = validate_rl_config({"algorithm": "gspo", "importance_sampling_level": "sequence", "loss_type": "grpo", "num_generations": 4, "max_completion_length": 96})
        self.assertEqual("token", grpo["importance_sampling_level"])
        self.assertEqual("sequence", gspo["importance_sampling_level"])

    def test_dr_grpo_cannot_be_called_gspo(self):
        with self.assertRaisesRegex(ValueError, "GSPO"):
            validate_rl_config({"algorithm": "gspo", "importance_sampling_level": "sequence", "loss_type": "dr_grpo", "num_generations": 4, "max_completion_length": 96})


if __name__ == "__main__":
    unittest.main()
