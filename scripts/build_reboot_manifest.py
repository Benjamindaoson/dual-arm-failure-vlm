from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.annotations import build_training_windows, load_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an auditable REBOOT VLM training manifest from meta/phase.json")
    parser.add_argument("--phase-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reboot_manifest.jsonl")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs" / "reboot_manifest_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--span-seconds", type=float, default=2.0)
    args = parser.parse_args()

    annotations = load_annotations(args.phase_json)
    windows, audit = build_training_windows(
        annotations, seed=args.seed, n_frames=args.frames, span_seconds=args.span_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in windows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    severe = {
        "phase_boundary_count_mismatch", "phase_boundaries_do_not_start_at_zero",
        "phase_boundaries_do_not_end_at_duration", "non_monotonic_phase_boundaries",
        "originating_phase_out_of_range", "failure_frame_out_of_range",
        "recovery_frame_out_of_range", "recovery_before_failure", "empty_failure_mode",
    }
    excluded = {k: v for k, v in audit.items() if severe.intersection(v)}
    report = {
        "dataset": annotations.dataset,
        "task_id": annotations.task_id,
        "episodes_total": len(annotations.episodes),
        "episodes_excluded": len(excluded),
        "windows_total": len(windows),
        "split_counts": dict(Counter(x.split for x in windows)),
        "state_counts": dict(Counter(x.execution_state for x in windows)),
        "failure_mode_counts": dict(Counter(x.failure_mode for x in windows)),
        "annotation_warnings": {k: v for k, v in audit.items() if v},
    }
    args.audit.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
