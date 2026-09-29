from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditResult:
    schema: dict[str, Any]
    annotation_audit: dict[str, Any]
    failure_mode_distribution: dict[str, Any]
    phase_distribution: dict[str, Any]


def audit_frame_payload(root: str, phase: dict[str, Any]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required for frame-table audit") from error

    from pathlib import Path

    base = Path(root)
    parquet_files = sorted((base / "data").rglob("*.parquet"))
    durations = {
        str(row["episode_index"]): int(row["duration_frames"])
        for row in phase.get("episodes", [])
        if isinstance(row, dict) and "episode_index" in row and "duration_frames" in row
    }
    width = max((len(key) for key in durations), default=1)
    counts: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()
    duplicate_rows = 0
    invalid_rows: list[dict[str, Any]] = []
    for path in parquet_files:
        file = parquet.ParquetFile(path)
        available = set(file.schema_arrow.names)
        required = {"episode_index", "frame_index"}
        if not required.issubset(available):
            invalid_rows.append({"file": str(path), "error": "missing_episode_or_frame_index"})
            continue
        for batch in file.iter_batches(columns=["episode_index", "frame_index"], batch_size=8192):
            for raw_episode, raw_frame in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
                episode_id = str(raw_episode).zfill(width)
                frame = int(raw_frame)
                key = (episode_id, frame)
                if key in seen:
                    duplicate_rows += 1
                seen.add(key)
                counts[episode_id] += 1
                duration = durations.get(episode_id)
                # In this snapshot `duration_frames` is the inclusive terminal
                # frame index: normal episodes contain frames 0..duration.
                if duration is None or not 0 <= frame <= duration:
                    if len(invalid_rows) < 100:
                        invalid_rows.append({"episode_index": episode_id, "frame_index": frame, "duration": duration})
    count_mismatches = {
        episode_id: {
            "observed_rows": counts.get(episode_id, 0),
            "expected_rows": duration + 1,
            "annotated_terminal_frame": duration,
        }
        for episode_id, duration in sorted(durations.items())
        if counts.get(episode_id, 0) != duration + 1
    }
    return {
        "parquet_files": len(parquet_files),
        "rows_observed": sum(counts.values()),
        "episodes_observed": len(counts),
        "duplicate_episode_frame_rows": duplicate_rows,
        "invalid_frame_rows": invalid_rows,
        "count_mismatches": count_mismatches,
        "duration_semantics": "inclusive_terminal_frame_index",
        "valid": bool(parquet_files) and duplicate_rows == 0 and not invalid_rows and not count_mismatches,
    }


def _shape(feature: dict[str, Any] | None) -> list[int] | None:
    if not isinstance(feature, dict) or not isinstance(feature.get("shape"), list):
        return None
    return [int(value) for value in feature["shape"]]


def _episode_issues(raw: Any, phase_count: int) -> list[str]:
    if not isinstance(raw, dict):
        return ["episode_not_object"]
    issues: list[str] = []
    required = ("episode_index", "task_id", "task_description", "episode_type", "duration_frames", "fps", "phase_boundaries", "phase_names", "failure")
    for key in required:
        if key not in raw:
            issues.append(f"missing_field:{key}")
    if issues:
        return issues

    names = raw["phase_names"]
    bounds = raw["phase_boundaries"]
    failure = raw["failure"]
    if not isinstance(names, list):
        issues.append("phase_names_not_array")
        names = []
    if len(names) != phase_count:
        issues.append(f"unexpected_phase_count:{len(names)}")
    if not isinstance(bounds, dict):
        issues.append("phase_boundaries_not_object")
        values: list[int] = []
    else:
        values = []
        for index in range(len(names) + 1):
            key = f"tau_{index}"
            if key not in bounds:
                issues.append(f"missing_field:phase_boundaries.{key}")
                continue
            try:
                values.append(int(bounds[key]))
            except (TypeError, ValueError):
                issues.append(f"invalid_integer:phase_boundaries.{key}")
    try:
        duration = int(raw["duration_frames"])
    except (TypeError, ValueError):
        duration = -1
        issues.append("invalid_integer:duration_frames")
    if values:
        if values[0] != 0:
            issues.append("phase_boundaries_do_not_start_at_zero")
        if values[-1] != duration:
            issues.append("phase_boundaries_do_not_end_at_duration")
        if any(left > right for left, right in zip(values, values[1:])):
            issues.append("non_monotonic_phase_boundaries")

    if not isinstance(failure, dict):
        issues.append("failure_not_object")
        return issues
    for key in ("originating_phase", "failure_mode", "induced_at_frame", "recovery_started_at_frame", "language_description"):
        if key not in failure:
            issues.append(f"missing_field:failure.{key}")
    if any(key not in failure for key in ("originating_phase", "failure_mode", "induced_at_frame", "recovery_started_at_frame")):
        return issues
    try:
        origin = int(failure["originating_phase"])
        induced = int(failure["induced_at_frame"])
        recovery = int(failure["recovery_started_at_frame"])
    except (TypeError, ValueError):
        issues.append("invalid_failure_integer")
        return issues
    # Official REBOOT material numbers the shared phases 1..5.
    if not 1 <= origin <= phase_count:
        issues.append("originating_phase_out_of_range")
    if not 0 <= induced < duration:
        issues.append("failure_frame_out_of_range")
    if not 0 <= recovery < duration:
        issues.append("recovery_frame_out_of_range")
    if recovery < induced:
        issues.append("recovery_before_failure")
    if not str(failure["failure_mode"]).strip():
        issues.append("empty_failure_mode")
    if not str(failure.get("language_description", "")).strip():
        issues.append("empty_recovery_description")
    return issues


def audit_dataset(info: dict[str, Any], phase: dict[str, Any], repository: dict[str, Any]) -> AuditResult:
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    camera_keys = sorted(key for key in features if key.startswith("observation.images."))
    rgb_keys = [key for key in camera_keys if not bool(features[key].get("info", {}).get("video.is_depth_map", False))]
    depth_keys = sorted(
        key for key, value in features.items()
        if isinstance(value, dict) and (value.get("dtype") == "depth" or bool(value.get("info", {}).get("video.is_depth_map", False)))
    )
    episodes = phase.get("episodes", []) if isinstance(phase.get("episodes"), list) else []
    phase_names = [str(value) for value in phase.get("phase_names", [])] if isinstance(phase.get("phase_names"), list) else []
    ids = [str(ep.get("episode_index", "<missing>")) for ep in episodes if isinstance(ep, dict)]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    issues_by_episode: dict[str, list[str]] = {}
    valid_ids: list[str] = []
    quarantined: list[str] = []
    annotated_duration = 0
    for row_number, episode in enumerate(episodes):
        episode_id = str(episode.get("episode_index", f"<row:{row_number}>") if isinstance(episode, dict) else f"<row:{row_number}>")
        issues = _episode_issues(episode, len(phase_names))
        if episode_id in duplicate_ids:
            issues.append("duplicate_episode")
        issues = sorted(set(issues))
        issues_by_episode[episode_id] = issues
        if isinstance(episode, dict):
            try:
                annotated_duration += int(episode.get("duration_frames", 0))
            except (TypeError, ValueError):
                pass
        if issues:
            quarantined.append(episode_id)
        else:
            valid_ids.append(episode_id)

    global_issues: list[str] = []
    if len(episodes) != int(phase.get("num_episodes", -1)) or len(episodes) != int(info.get("total_episodes", -1)):
        global_issues.append("episode_count_mismatch")
    expected_rows = annotated_duration + len(episodes)
    if expected_rows != int(info.get("total_frames", -1)):
        global_issues.append("annotated_duration_sum_mismatch")
    if str(phase.get("dataset", "")) != str(repository.get("id", "")):
        global_issues.append("dataset_identifier_mismatch")

    issue_counts = Counter(issue for issues in issues_by_episode.values() for issue in issues)
    all_modes = Counter()
    valid_modes = Counter()
    all_phases = Counter()
    valid_phases = Counter()
    valid_id_set = set(valid_ids)
    for episode in episodes:
        if not isinstance(episode, dict) or not isinstance(episode.get("failure"), dict):
            continue
        failure = episode["failure"]
        mode = str(failure.get("failure_mode", "<missing>"))
        all_modes[mode] += 1
        try:
            origin = int(failure.get("originating_phase"))
            phase_label = phase_names[origin - 1] if 1 <= origin <= len(phase_names) else f"__invalid_{origin}__"
        except (TypeError, ValueError):
            phase_label = "__invalid__"
        all_phases[phase_label] += 1
        if str(episode.get("episode_index")) in valid_id_set:
            valid_modes[mode] += 1
            valid_phases[phase_label] += 1

    schema = {
        "dataset_repo": repository.get("id"),
        "dataset_revision": repository.get("sha"),
        "last_modified": repository.get("lastModified"),
        "license": (repository.get("cardData") or {}).get("license") if isinstance(repository.get("cardData"), dict) else None,
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "camera_keys": camera_keys,
        "rgb_camera_keys": rgb_keys,
        "depth_keys": depth_keys,
        "has_rgb": bool(rgb_keys),
        "has_depth": bool(depth_keys),
        "robot_state": {"dtype": (features.get("observation.state") or {}).get("dtype"), "shape": _shape(features.get("observation.state"))},
        "action": {"dtype": (features.get("action") or {}).get("dtype"), "shape": _shape(features.get("action"))},
        "episode_index": {"frame_table_dtype": (features.get("episode_index") or {}).get("dtype"), "annotation_type": type(episodes[0].get("episode_index")).__name__ if episodes and isinstance(episodes[0], dict) else None},
        "phase_names": phase_names,
        "phase_annotation_fields": ["phase_names", "phase_boundaries", "originating_phase", "failure_mode", "induced_at_frame", "recovery_started_at_frame", "language_description"],
    }
    annotation_audit = {
        "episodes_declared": phase.get("num_episodes"),
        "episodes_observed": len(episodes),
        "episodes_valid": len(valid_ids),
        "episodes_quarantined": len(quarantined),
        "valid_episode_ids": sorted(valid_ids),
        "quarantined_episode_ids": sorted(quarantined),
        "duplicate_episode_ids": duplicate_ids,
        "issues_by_episode": {key: value for key, value in sorted(issues_by_episode.items()) if value},
        "issue_counts": dict(sorted(issue_counts.items())),
        "global_issues": global_issues,
        "annotated_duration_frames": annotated_duration,
        "expected_rows_from_inclusive_duration": expected_rows,
        "info_total_frames": info.get("total_frames"),
        "originating_phase_convention": "one_based_1_to_phase_count",
    }
    return AuditResult(
        schema=schema,
        annotation_audit=annotation_audit,
        failure_mode_distribution={"all_annotations": dict(sorted(all_modes.items())), "valid_episodes": dict(sorted(valid_modes.items()))},
        phase_distribution={"all_annotations": dict(sorted(all_phases.items())), "valid_episodes": dict(sorted(valid_phases.items()))},
    )
