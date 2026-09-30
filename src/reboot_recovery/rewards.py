from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


PHASE_LABELS = {
    "align (pick)",
    "engage (pick)",
    "transport",
    "align (place)",
    "engage (place)",
}

FAILURE_MODE_LABELS = {
    "delayed_start",
    "excessive_force",
    "freeze",
    "jamming",
    "misalignment",
    "poor_engagement",
    "premature_grasp",
    "premature_release",
    "retention_failure",
    "slip",
    "sub_mm_misalignment",
}


@dataclass(frozen=True)
class RewardBreakdown:
    valid_json: float
    phase: float
    state: float
    failure_mode: float
    total: float


def parse_structured_prediction(
    text: str,
    *,
    phase_labels: set[str] | None = None,
    failure_mode_labels: set[str] | None = None,
) -> dict[str, str] | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    required = {"phase", "state", "failure_mode"}
    if set(value) != required or any(not isinstance(value[key], str) for key in required):
        return None
    parsed = {key: value[key].strip().casefold() for key in required}
    if not all(parsed.values()):
        return None
    phases = PHASE_LABELS if phase_labels is None else {label.casefold() for label in phase_labels}
    failure_modes = FAILURE_MODE_LABELS if failure_mode_labels is None else {
        label.casefold() for label in failure_mode_labels
    }
    if parsed["phase"] not in phases or parsed["state"] not in {"nominal", "failure", "recovery"}:
        return None
    if parsed["state"] == "nominal":
        if parsed["failure_mode"] != "none":
            return None
    elif parsed["failure_mode"] not in failure_modes:
        return None
    return parsed


def score_prediction(
    reference: dict[str, Any],
    output: str,
    *,
    phase_weight: float = 0.25,
    state_weight: float = 0.35,
    failure_weight: float = 0.40,
    label_weight: float = 1.0,
    phase_labels: set[str] | None = None,
    failure_mode_labels: set[str] | None = None,
) -> RewardBreakdown:
    """Task reward for RLVR-style training.

    Formatting is a gate, not a positive shortcut. For nominal windows the
    failure-mode field is excluded because `none` is bookkeeping, not a
    learned failure category.
    """
    pred = parse_structured_prediction(
        output,
        phase_labels=phase_labels,
        failure_mode_labels=failure_mode_labels,
    )
    if pred is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)

    ref_phase = str(reference["phase_name"]).strip().casefold()
    ref_state = str(reference["execution_state"]).strip().casefold()
    ref_mode = str(reference["failure_mode"]).strip().casefold()
    if min(phase_weight, state_weight, failure_weight, label_weight) < 0:
        raise ValueError("reward weights must be non-negative")
    phase = 1.0 if pred["phase"] == ref_phase else 0.0
    state = 1.0 if pred["state"] == ref_state else 0.0
    if ref_state == "nominal":
        failure_mode = 0.0
        total = (
            (phase_weight * phase + state_weight * state) / (phase_weight + state_weight)
            if phase_weight + state_weight else 0.0
        )
    else:
        failure_mode = 1.0 if pred["failure_mode"] == ref_mode else 0.0
        denominator = phase_weight + state_weight + failure_weight
        total = (
            (phase_weight * phase + state_weight * state + failure_weight * failure_mode) / denominator
            if denominator else 0.0
        )
    return RewardBreakdown(1.0, phase, state, failure_mode, total * label_weight)


def score_state_gated_prediction(
    reference: dict[str, Any], output: str, *,
    phase_labels: set[str] | None = None, failure_mode_labels: set[str] | None = None,
) -> float:
    pred = parse_structured_prediction(
        output, phase_labels=phase_labels, failure_mode_labels=failure_mode_labels,
    )
    if pred is None:
        return 0.0
    state = str(reference["execution_state"]).strip().casefold()
    if pred["state"] != state:
        return 0.0
    phase = pred["phase"] == str(reference["phase_name"]).strip().casefold()
    if state == "nominal":
        return 0.7 + 0.3 * phase
    mode = pred["failure_mode"] == str(reference["failure_mode"]).strip().casefold()
    # ponytail: no negative false-negative term; add it only if this gate still collapses.
    return 0.7 + 0.1 * phase + 0.2 * mode
