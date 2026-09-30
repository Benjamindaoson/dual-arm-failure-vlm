from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Iterable

from .annotations import EpisodeAnnotation, state_anchor_frames


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split_receipt(
    episodes: Iterable[EpisodeAnnotation],
    assignments: dict[str, str],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, object]:
    rows = sorted(episodes, key=lambda episode: episode.episode_index)
    expected = {episode.episode_index for episode in rows}
    if set(assignments) != expected:
        raise ValueError("split assignments must contain every episode exactly once")
    split_names = ("train", "val", "test")
    episode_ids = {name: sorted(key for key, value in assignments.items() if value == name) for name in split_names}
    if sum(len(value) for value in episode_ids.values()) != len(expected):
        raise ValueError("unknown split name or duplicate assignment")
    distributions: dict[str, object] = {}
    for split in split_names:
        selected = [episode for episode in rows if assignments[episode.episode_index] == split]
        distributions[split] = {
            "episodes": len(selected),
            "state_windows": dict(sorted(Counter(
                state
                for ep in selected
                for state, frames in state_anchor_frames(ep).items()
                for _ in frames
            ).items())),
            "failure_modes": dict(sorted(Counter(ep.failure.failure_mode for ep in selected).items())),
            "originating_phases": dict(sorted(Counter(ep.phase_names[ep.failure.originating_phase - 1] for ep in selected).items())),
        }
    payload: dict[str, object] = {
        "unit": "episode",
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio},
        "episodes": episode_ids,
        "distributions": distributions,
        "leakage_detected": False,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload
