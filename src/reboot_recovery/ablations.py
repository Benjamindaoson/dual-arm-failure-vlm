from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class AblationSpec:
    name: str
    num_frames: int
    cameras: tuple[str, ...]
    include_trace: bool


DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
)


def resolve_ablation(name: str, *, num_frames: int = 4, cameras: Sequence[str] = DEFAULT_CAMERAS) -> AblationSpec:
    key = name.upper()
    if key == "A0":
        return AblationSpec(key, 1, (cameras[0],), False)
    if key == "A1":
        return AblationSpec(key, num_frames, (cameras[0],), False)
    if key == "A2":
        return AblationSpec(key, num_frames, tuple(cameras[:2]), False)
    if key == "A3":
        return AblationSpec(key, num_frames, tuple(cameras[:2]), True)
    raise ValueError("ablation must be A0, A1, A2, or A3")


def select_window_frames(row: dict[str, Any], spec: AblationSpec) -> list[int]:
    if spec.name == "A0":
        return [int(row["anchor_frame"])]
    available = [int(value) for value in row["sampled_frames"]]
    if len(available) < spec.num_frames:
        raise ValueError(f"manifest has {len(available)} frames but {spec.num_frames} are required")
    if len(available) == spec.num_frames:
        return available
    return [available[round(index * (len(available) - 1) / (spec.num_frames - 1))] for index in range(spec.num_frames)]


def _vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [float(item) for item in value]


def summarize_trace(states: Iterable[Any], actions: Iterable[Any]) -> str:
    state_rows = [_vector(value) for value in states]
    action_rows = [_vector(value) for value in actions]
    if not state_rows or not action_rows:
        raise ValueError("state and action trace cannot be empty")
    if len({len(row) for row in state_rows}) != 1 or len({len(row) for row in action_rows}) != 1:
        raise ValueError("trace vectors must have stable dimensions")
    state_delta = [last - first for first, last in zip(state_rows[0], state_rows[-1], strict=True)]
    action_last = action_rows[-1]
    fmt = lambda values: "[" + ",".join(f"{value:.4f}" for value in values) + "]"
    return (
        f"steps={len(state_rows)}; state_dim={len(state_rows[0])}; action_dim={len(action_last)}; "
        f"state_delta={fmt(state_delta)}; latest_action={fmt(action_last)}"
    )
