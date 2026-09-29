from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class RewardBreakdown:
    valid_json: float
    phase: float
    state: float
    failure_mode: float
    total: float


def parse_structured_prediction(text: str) -> dict[str, str] | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    required = ("phase", "state", "failure_mode")
    if any(k not in value for k in required):
        return None
    return {k: str(value[k]).strip().casefold() for k in required}


def score_prediction(reference: dict[str, Any], output: str) -> RewardBreakdown:
    """Task reward for RLVR-style training.

    Formatting is a gate, not a positive shortcut. For nominal windows the
    failure-mode field is excluded because `none` is bookkeeping, not a
    learned failure category.
    """
    pred = parse_structured_prediction(output)
    if pred is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)

    ref_phase = str(reference["phase_name"]).strip().casefold()
    ref_state = str(reference["execution_state"]).strip().casefold()
    ref_mode = str(reference["failure_mode"]).strip().casefold()
    phase = 1.0 if pred["phase"] == ref_phase else 0.0
    state = 1.0 if pred["state"] == ref_state else 0.0
    if ref_state == "nominal":
        failure_mode = 0.0
        total = (phase + state) / 2.0
    else:
        failure_mode = 1.0 if pred["failure_mode"] == ref_mode else 0.0
        total = (phase + state + failure_mode) / 3.0
    return RewardBreakdown(1.0, phase, state, failure_mode, total)
