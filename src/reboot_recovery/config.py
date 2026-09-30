from __future__ import annotations

from typing import Any, Mapping


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
    if reward_scheme == "state_gated_v2":
        if algorithm != "grpo":
            raise ValueError("V2 state-gated reward is authorized for GRPO only")
        if value.get("reward_weights") != [1.0]:
            raise ValueError("state_gated_v2 requires reward_weights=[1.0]")
    value.update({"algorithm": algorithm, "importance_sampling_level": importance, "loss_type": loss_type,
                  "num_generations": generations, "max_completion_length": completion,
                  "reward_scheme": reward_scheme})
    return value
