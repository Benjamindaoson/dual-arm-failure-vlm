from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

EXPECTED_PHASE_COUNT = 5


@dataclass(frozen=True)
class FailureAnnotation:
    originating_phase: int
    failure_mode: str
    induced_at_frame: int
    recovery_started_at_frame: int
    language_description: str


@dataclass(frozen=True)
class EpisodeAnnotation:
    episode_index: str
    task_id: str
    task_description: str
    episode_type: str
    duration_frames: int
    fps: int
    phase_boundaries: tuple[int, ...]
    phase_names: tuple[str, ...]
    failure: FailureAnnotation
    operator_id: str | None = None
    collection_session: str | None = None

    def phase_at(self, frame: int) -> int:
        if not 0 <= frame < self.duration_frames:
            raise ValueError(f"frame {frame} outside [0, {self.duration_frames})")
        for phase_idx in range(len(self.phase_names)):
            if self.phase_boundaries[phase_idx] <= frame < self.phase_boundaries[phase_idx + 1]:
                return phase_idx
        return len(self.phase_names) - 1

    def state_at(self, frame: int) -> str:
        """Return an annotation-grounded execution state without inventing success labels."""
        if frame < self.failure.induced_at_frame:
            return "nominal"
        if frame < self.failure.recovery_started_at_frame:
            return "failure"
        return "recovery"


@dataclass(frozen=True)
class RebootAnnotations:
    dataset: str
    task_id: str
    task_description: str
    fps: int
    phase_names: tuple[str, ...]
    episodes: tuple[EpisodeAnnotation, ...]


@dataclass(frozen=True)
class TrainingWindow:
    episode_index: str
    split: str
    anchor_kind: str
    anchor_frame: int
    sampled_frames: tuple[int, ...]
    phase_index: int
    phase_name: str
    execution_state: str
    failure_mode: str
    task_id: str
    task_description: str
    recovery_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "split": self.split,
            "anchor_kind": self.anchor_kind,
            "anchor_frame": self.anchor_frame,
            "sampled_frames": list(self.sampled_frames),
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "execution_state": self.execution_state,
            "failure_mode": self.failure_mode,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "recovery_description": self.recovery_description,
        }


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def _parse_episode(raw: dict[str, Any]) -> EpisodeAnnotation:
    phase_names = tuple(str(x) for x in _require(raw, "phase_names"))
    bounds_raw = _require(raw, "phase_boundaries")
    boundaries = tuple(int(_require(bounds_raw, f"tau_{i}")) for i in range(len(phase_names) + 1))
    fail_raw = _require(raw, "failure")
    failure = FailureAnnotation(
        originating_phase=int(_require(fail_raw, "originating_phase")),
        failure_mode=str(_require(fail_raw, "failure_mode")).strip(),
        induced_at_frame=int(_require(fail_raw, "induced_at_frame")),
        recovery_started_at_frame=int(_require(fail_raw, "recovery_started_at_frame")),
        language_description=str(fail_raw.get("language_description", "")).strip(),
    )
    return EpisodeAnnotation(
        episode_index=str(_require(raw, "episode_index")),
        task_id=str(_require(raw, "task_id")),
        task_description=str(_require(raw, "task_description")),
        episode_type=str(_require(raw, "episode_type")),
        duration_frames=int(_require(raw, "duration_frames")),
        fps=int(_require(raw, "fps")),
        phase_boundaries=boundaries,
        phase_names=phase_names,
        failure=failure,
        operator_id=(str(raw["operator_id"]) if raw.get("operator_id") is not None else None),
        collection_session=(str(raw["collection_session"]) if raw.get("collection_session") is not None else None),
    )


def load_annotations(path: str | Path) -> RebootAnnotations:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    phase_names = tuple(str(x) for x in _require(raw, "phase_names"))
    episodes = tuple(_parse_episode(x) for x in _require(raw, "episodes"))
    return RebootAnnotations(
        dataset=str(_require(raw, "dataset")),
        task_id=str(_require(raw, "task_id")),
        task_description=str(_require(raw, "task_description")),
        fps=int(_require(raw, "fps")),
        phase_names=phase_names,
        episodes=episodes,
    )


def validate_episode(ep: EpisodeAnnotation) -> list[str]:
    warnings: list[str] = []
    if len(ep.phase_names) != EXPECTED_PHASE_COUNT:
        warnings.append(f"unexpected_phase_count:{len(ep.phase_names)}")
    if len(ep.phase_boundaries) != len(ep.phase_names) + 1:
        warnings.append("phase_boundary_count_mismatch")
    if ep.phase_boundaries and ep.phase_boundaries[0] != 0:
        warnings.append("phase_boundaries_do_not_start_at_zero")
    if ep.phase_boundaries and ep.phase_boundaries[-1] != ep.duration_frames:
        warnings.append("phase_boundaries_do_not_end_at_duration")
    if any(a > b for a, b in zip(ep.phase_boundaries, ep.phase_boundaries[1:])):
        warnings.append("non_monotonic_phase_boundaries")
    f = ep.failure
    if not 0 <= f.originating_phase < len(ep.phase_names):
        warnings.append("originating_phase_out_of_range")
    if not 0 <= f.induced_at_frame < ep.duration_frames:
        warnings.append("failure_frame_out_of_range")
    if not 0 <= f.recovery_started_at_frame < ep.duration_frames:
        warnings.append("recovery_frame_out_of_range")
    if f.recovery_started_at_frame < f.induced_at_frame:
        warnings.append("recovery_before_failure")
    if not f.failure_mode:
        warnings.append("empty_failure_mode")
    return warnings


def usable_episode(ep: EpisodeAnnotation) -> bool:
    severe = {
        "phase_boundary_count_mismatch",
        "phase_boundaries_do_not_start_at_zero",
        "phase_boundaries_do_not_end_at_duration",
        "non_monotonic_phase_boundaries",
        "originating_phase_out_of_range",
        "failure_frame_out_of_range",
        "recovery_frame_out_of_range",
        "recovery_before_failure",
        "empty_failure_mode",
    }
    return not severe.intersection(validate_episode(ep))


def _sample_frames(center: int, duration: int, fps: int, n_frames: int, span_seconds: float) -> tuple[int, ...]:
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if span_seconds <= 0:
        raise ValueError("span_seconds must be > 0")
    half_span = max(1, int(round(fps * span_seconds / 2)))
    lo = max(0, center - half_span)
    hi = min(duration - 1, center + half_span)
    if n_frames == 1 or hi == lo:
        return (min(max(center, 0), duration - 1),)
    return tuple(round(lo + i * (hi - lo) / (n_frames - 1)) for i in range(n_frames))


def split_episode_ids(
    episodes: Iterable[EpisodeAnnotation],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, str]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")
    ids = [ep.episode_index for ep in episodes]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    split: dict[str, str] = {}
    for i, episode_id in enumerate(ids):
        split[episode_id] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return split


def build_training_windows(
    annotations: RebootAnnotations,
    *,
    seed: int = 42,
    n_frames: int = 6,
    span_seconds: float = 2.0,
    min_gap_frames: int = 15,
) -> tuple[list[TrainingWindow], dict[str, list[str]]]:
    """Build three balanced, episode-disjoint anchors: nominal, failure, recovery.

    Annotation-inconsistent episodes are excluded rather than silently repaired.
    """
    audit = {ep.episode_index: validate_episode(ep) for ep in annotations.episodes}
    usable = [ep for ep in annotations.episodes if usable_episode(ep)]
    split = split_episode_ids(usable, seed=seed)
    windows: list[TrainingWindow] = []

    for ep in usable:
        f = ep.failure
        anchors: list[tuple[str, int]] = []
        nominal_center = max(0, f.induced_at_frame - max(min_gap_frames, ep.fps))
        anchors.append(("nominal", nominal_center))
        if f.recovery_started_at_frame > f.induced_at_frame:
            anchors.append(("failure", (f.induced_at_frame + f.recovery_started_at_frame) // 2))
        recovery_center = min(ep.duration_frames - 1, f.recovery_started_at_frame + max(min_gap_frames, ep.fps))
        anchors.append(("recovery", recovery_center))

        for anchor_kind, center in anchors:
            phase_idx = ep.phase_at(center)
            state = ep.state_at(center)
            windows.append(
                TrainingWindow(
                    episode_index=ep.episode_index,
                    split=split[ep.episode_index],
                    anchor_kind=anchor_kind,
                    anchor_frame=center,
                    sampled_frames=_sample_frames(center, ep.duration_frames, ep.fps, n_frames, span_seconds),
                    phase_index=phase_idx,
                    phase_name=ep.phase_names[phase_idx],
                    execution_state=state,
                    failure_mode=(f.failure_mode if state != "nominal" else "none"),
                    task_id=ep.task_id,
                    task_description=ep.task_description,
                    recovery_description=f.language_description,
                )
            )
    return windows, audit
