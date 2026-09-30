from __future__ import annotations

from typing import Any, Mapping


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
