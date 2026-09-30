from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

from .annotations import TrainingWindow


def to_state_only_record(row: dict[str, Any]) -> dict[str, Any]:
    reference = row["reference"]
    trace = row.get("trace_summary")
    trace_line = f"Robot trace summary:\n{trace}\n" if trace else ""
    prompt = (
        "You are a robot execution critic for precision assembly. "
        "Use only the ordered visual observations and task instruction to identify the current execution state.\n"
        f"Task: {reference['task_description']}\n"
        f"{trace_line}"
        "Return JSON only with exactly one key: state. "
        "state must be one of: nominal, failure, recovery."
    )
    return {
        **row,
        "prompt": [{"role": "user", "content": [{"type": "image"} for _ in row["images"]] + [{"type": "text", "text": prompt}]}],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": json.dumps(
            {"state": reference["execution_state"]}, separators=(",", ":")
        )}]}],
    }


def build_prompt(
    window: TrainingWindow,
    *,
    include_trace_summary: bool = False,
    trace_summary: str | None = None,
    phase_names: Sequence[str] = ("Align (pick)", "Engage (pick)", "Transport", "Align (place)", "Engage (place)"),
    failure_modes: Sequence[str] = (),
) -> str:
    trace_block = ""
    if include_trace_summary:
        if not trace_summary:
            raise ValueError("trace_summary is required when include_trace_summary=True")
        trace_block = f"\nRobot trace summary:\n{trace_summary}\n"
    mode_line = (
        f"failure_mode must be one of: none, {', '.join(failure_modes)}.\n"
        if failure_modes else "failure_mode must come from the dataset taxonomy.\n"
    )
    return (
        "You are a robot execution critic for precision assembly. "
        "Use the ordered visual observations and the task instruction to diagnose the current execution state.\n"
        f"Task: {window.task_description}\n"
        f"{trace_block}"
        "Return JSON only with exactly these keys: phase, state, failure_mode.\n"
        f"phase must be one of: {', '.join(phase_names)}.\n"
        "state must be one of: nominal, failure, recovery.\n"
        f"{mode_line}"
        "If state is nominal, failure_mode must be none. Otherwise predict the observed failure mode."
    )
