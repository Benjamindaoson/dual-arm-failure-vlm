from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a paired Base/SFT failure-timing table")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {"base": read_object(args.base), "sft": read_object(args.sft)}
    episode_ids = {name: set(run["episodes"]) for name, run in runs.items()}
    if episode_ids["base"] != episode_ids["sft"]:
        raise SystemExit("Base and SFT timing episodes differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "episode_index", "detection_delay_seconds", "false_alarms_before_onset", "windows"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, run in runs.items():
            for episode_id, values in sorted(run["episodes"].items()):
                writer.writerow({
                    "model": model,
                    "episode_index": episode_id,
                    "detection_delay_seconds": values["detection_delay_seconds"],
                    "false_alarms_before_onset": values["false_alarms_before_onset"],
                    "windows": values["windows"],
                })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
