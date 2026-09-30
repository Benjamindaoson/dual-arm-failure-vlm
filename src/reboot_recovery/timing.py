from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


def timing_slice(relative_seconds: float, recovery_relative_seconds: float | None = None) -> str:
    if relative_seconds < 0:
        return "pre_failure"
    if recovery_relative_seconds is not None and relative_seconds >= recovery_relative_seconds:
        return "early_recovery" if relative_seconds < recovery_relative_seconds + 1.0 else "late_recovery"
    return "early_failure" if relative_seconds <= 1.0 else "late_failure"


def evaluate_failure_timing(
    rows: Iterable[dict[str, Any]], *, consecutive: int = 2, target_state: str = "failure"
) -> dict[str, Any]:
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    target_state = target_state.casefold()
    if target_state not in {"failure", "recovery"}:
        raise ValueError("target_state must be failure or recovery")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_index"])].append(row)
    episodes: dict[str, dict[str, Any]] = {}
    detected_delays: list[float] = []
    for episode_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: float(item["relative_seconds"]))
        false_alarms = sum(
            float(item["relative_seconds"]) < 0
            and str(item["predicted_state"]).casefold() == target_state
            for item in ordered
        )
        pre_onset_windows = sum(float(item["relative_seconds"]) < 0 for item in ordered)
        post = [item for item in ordered if float(item["relative_seconds"]) >= 0]
        delay: float | None = None
        for index in range(consecutive - 1, len(post)):
            run = post[index - consecutive + 1:index + 1]
            if all(
                str(item.get("gold_state", target_state)).casefold() == target_state
                and str(item["predicted_state"]).casefold() == target_state
                for item in run
            ):
                delay = float(post[index]["relative_seconds"])
                detected_delays.append(delay)
                break
        episode = {
            "detection_delay_seconds": delay,
            "false_alarms_before_onset": int(false_alarms),
            "pre_onset_windows": pre_onset_windows,
            "pre_onset_false_alarm_rate": false_alarms / pre_onset_windows if pre_onset_windows else None,
            "windows": len(ordered),
        }
        if target_state == "failure":
            episode["false_alarms_before_failure"] = int(false_alarms)
        episodes[episode_id] = episode
    result = {
        "target_state": target_state,
        "consecutive_windows_required": consecutive,
        "episode_count": len(episodes),
        "detected_episodes": len(detected_delays),
        "undetected_episodes": len(episodes) - len(detected_delays),
        "mean_detection_delay_seconds": mean(detected_delays) if detected_delays else None,
        "median_detection_delay_seconds": median(detected_delays) if detected_delays else None,
        "false_alarms_before_onset": sum(item["false_alarms_before_onset"] for item in episodes.values()),
        "pre_onset_windows": sum(item["pre_onset_windows"] for item in episodes.values()),
        "episodes": episodes,
    }
    if target_state == "failure":
        result["false_alarms_before_failure"] = result["false_alarms_before_onset"]
    result["pre_onset_false_alarm_rate"] = (
        result["false_alarms_before_onset"] / result["pre_onset_windows"]
        if result["pre_onset_windows"] else None
    )
    return result
