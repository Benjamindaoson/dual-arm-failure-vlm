from __future__ import annotations

from typing import Any, Mapping

from .rewards import DEFAULT_STATE_GATED_REWARD


def validate_rl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(config)
    algorithm = str(value.get("algorithm", "")).casefold()
    importance = str(value.get("importance_sampling_level", "")).casefold()
    loss_type = str(value.get("loss_type", "")).casefold()
    expected = {"grpo": "token", "gspo": "sequence"}
    if algorithm not in expected:
        raise ValueError("algorithm must be grpo or gspo")
    if importance != expected[algorithm]:
        raise ValueError(f"{algorithm.upper()} requires importance_sampling_level={expected[algorithm]}")
    if loss_type != "grpo":
        if algorithm == "gspo":
            raise ValueError("GSPO requires loss_type=grpo; sequence + dr_grpo is not paper GSPO")
        raise ValueError("GRPO baseline requires loss_type=grpo")
    generations = int(value.get("num_generations", 0))
    completion = int(value.get("max_completion_length", 0))
    if generations < 2:
        raise ValueError("num_generations must be >= 2")
    if not 1 <= completion <= 256:
        raise ValueError("max_completion_length must be between 1 and 256 for structured diagnosis")
    reward_scheme = str(value.get("reward_scheme", "additive_v1"))
    if reward_scheme not in {"additive_v1", "state_gated_v2"}:
        raise ValueError("unknown reward_scheme")
    if value.get("class_weighting", "inverse_sqrt") not in {"none", "inverse_sqrt"}:
        raise ValueError("class_weighting must be none or inverse_sqrt")
    if reward_scheme == "state_gated_v2":
        if value.get("reward_weights") != [1.0]:
            raise ValueError("state_gated_v2 requires reward_weights=[1.0]")
        reward_config = value.get("state_gated_reward", DEFAULT_STATE_GATED_REWARD)
        if not isinstance(reward_config, Mapping) or set(reward_config) != set(DEFAULT_STATE_GATED_REWARD):
            raise ValueError("state_gated_reward requires exactly four configured components")
        if float(reward_config["failure_miss_penalty"]) >= 0 or any(
            float(reward_config[key]) < 0 for key in ("correct_state_base", "phase_bonus", "failure_mode_bonus")
        ):
            raise ValueError("state-gated reward penalty must be negative and bonuses nonnegative")
        value["state_gated_reward"] = {key: float(reward_config[key]) for key in DEFAULT_STATE_GATED_REWARD}
    value.update({"algorithm": algorithm, "importance_sampling_level": importance, "loss_type": loss_type,
                  "num_generations": generations, "max_completion_length": completion,
                  "reward_scheme": reward_scheme})
    return value
