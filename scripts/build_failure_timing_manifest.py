from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.annotations import causal_sample_frames, load_annotations, usable_episode
from reboot_recovery.timing import timing_slice


OFFSETS_SECONDS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build causal failure-onset windows for held-out REBOOT episodes")
    parser.add_argument("--phase-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--span-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "eval" / "failure_timing_manifest.jsonl")
    args = parser.parse_args(argv)

    annotations = load_annotations(args.phase_json)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    allowed = set(split["episodes"][args.split])
    rows = []
    for episode in annotations.episodes:
        if episode.episode_index not in allowed or not usable_episode(episode):
            continue
        recovery_relative = (episode.failure.recovery_started_at_frame - episode.failure.induced_at_frame) / episode.fps
        events = {
            "failure_onset": episode.failure.induced_at_frame,
            "recovery_onset": episode.failure.recovery_started_at_frame,
        }
        for event_kind, event_frame in events.items():
            for offset in OFFSETS_SECONDS:
                anchor = min(episode.duration_frames - 1, max(0, round(event_frame + offset * episode.fps)))
                phase_index = episode.phase_at(anchor)
                state = episode.state_at(anchor)
                rows.append({
                    "id": f"ep{episode.episode_index}-{event_kind}-{offset:+.1f}s",
                    "episode_index": episode.episode_index,
                    "split": args.split,
                    "anchor_frame": anchor,
                    "relative_seconds": offset,
                    "event_kind": event_kind,
                    "event_frame": event_frame,
                    "recovery_relative_seconds": recovery_relative,
                    "timing_slice": timing_slice(offset, recovery_relative) if event_kind == "failure_onset" else f"recovery_{offset:+.1f}s",
                    "sampled_frames": list(causal_sample_frames(anchor, episode.duration_frames, episode.fps, args.num_frames, args.span_seconds)),
                    "phase_index": phase_index,
                    "phase_name": episode.phase_names[phase_index],
                    "execution_state": state,
                    "failure_mode": episode.failure.failure_mode if state != "nominal" else "none",
                    "task_id": episode.task_id,
                    "task_description": episode.task_description,
                    "induced_at_frame": episode.failure.induced_at_frame,
                    "recovery_started_at_frame": episode.failure.recovery_started_at_frame,
                    "fps": episode.fps,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "episodes": len({row['episode_index'] for row in rows}), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
