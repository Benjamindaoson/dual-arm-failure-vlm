from __future__ import annotations

from typing import Any, Mapping


def decide_v2_rl_gate(
    base: Mapping[str, Any], sft: Mapping[str, Any], *, verifier_validated: bool
) -> dict[str, Any]:
    if base.get("task_schema") != "full" or sft.get("task_schema") != "full":
        raise ValueError("V2 RL gate requires matching full-schema evaluations")
    base_support = int(base["failure_support"])
    sft_support = int(sft["failure_support"])
    base_correct = int(base["failure_correct"])
    sft_correct = int(sft["failure_correct"])
    if base_support <= 0 or sft_support != base_support:
        raise ValueError("V2 gate requires matching nonzero failure support")
    if not (0 <= base_correct <= base_support and 0 <= sft_correct <= sft_support):
        raise ValueError("failure correct counts must be within support")
    state_gain = float(sft["state_macro_f1"]) - float(base["state_macro_f1"])
    false_alarm_gain = float(sft.get("pre_failure_false_positive_rate", 0.0)) - float(base.get("pre_failure_false_positive_rate", 0.0))
    compliance = float(sft.get("strict_json_valid_rate", sft.get("json_valid_rate", 0.0)))
    if state_gain < 0 or false_alarm_gain > 0 or compliance < 0.95:
        decision = "REVISIT_REPRESENTATION_OR_SUPERVISION"
    elif sft_correct == sft_support:
        decision = "NO_OUTCOME_HEADROOM"
    elif sft_correct <= base_correct:
        decision = "REVISIT_REPRESENTATION_OR_SUPERVISION"
    elif not verifier_validated:
        decision = "VERIFY_REWARD_FIRST"
    else:
        decision = "RUN_RLVR"
    return {
        "protocol": "V2",
        "task_schema": "full",
        "decision": decision,
        "base_failure_correct": base_correct,
        "sft_failure_correct": sft_correct,
        "failure_support": sft_support,
        "failure_correct_gain": sft_correct - base_correct,
        "strict_json_valid_rate": compliance,
        "state_macro_f1_gain": state_gain,
        "pre_failure_false_positive_rate_gain": false_alarm_gain,
        "verifier_validated": verifier_validated,
    }


def decide_stage(
    base: Mapping[str, Any],
    sft: Mapping[str, Any] | None = None,
    *,
    target_failure_recall: float = 0.90,
    target_state_macro_f1: float = 0.85,
    min_state_gain: float = 0.03,
    min_failure_gain: float = 0.03,
) -> dict[str, Any]:
    base_failure = float(base.get("failure_recall", 0.0))
    base_state = float(base.get("state_macro_f1", 0.0))
    if sft is None:
        sufficient = base_failure >= target_failure_recall and base_state >= target_state_macro_f1
        return {
            "decision": "BASE_SUFFICIENT" if sufficient else "RUN_SFT",
            "base_failure_recall": base_failure,
            "base_state_macro_f1": base_state,
            "thresholds": {"failure_recall": target_failure_recall, "state_macro_f1": target_state_macro_f1},
        }
    sft_failure = float(sft.get("failure_recall", 0.0))
    sft_state = float(sft.get("state_macro_f1", 0.0))
    sufficient = sft_failure >= target_failure_recall and sft_state >= target_state_macro_f1
    learned = (sft_failure - base_failure >= min_failure_gain or sft_state - base_state >= min_state_gain) and sft_failure >= base_failure and sft_state >= base_state
    decision = "SFT_SUFFICIENT" if sufficient else ("RUN_RLVR" if learned else "REVISIT_DATA_OR_TASK")
    return {
        "decision": decision,
        "base_failure_recall": base_failure,
        "base_state_macro_f1": base_state,
        "sft_failure_recall": sft_failure,
        "sft_state_macro_f1": sft_state,
        "gains": {"failure_recall": sft_failure - base_failure, "state_macro_f1": sft_state - base_state},
        "thresholds": {
            "failure_recall": target_failure_recall,
            "state_macro_f1": target_state_macro_f1,
            "min_failure_gain": min_failure_gain,
            "min_state_gain": min_state_gain,
        },
    }
