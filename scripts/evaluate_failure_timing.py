from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.rewards import parse_structured_prediction
from reboot_recovery.timing import evaluate_failure_timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure sustained failure-detection delay and pre-onset false alarms")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--event-kind", default="failure_onset")
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "eval" / "failure_timing.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "artifacts" / "eval" / "failure_timing.csv")
    args = parser.parse_args(argv)
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        reference = prediction["reference"]
        if reference.get("event_kind", "failure_onset") != args.event_kind:
            continue
        parsed = parse_structured_prediction(str(prediction.get("output", "")))
        rows.append({
            "episode_index": reference["episode_index"],
            "relative_seconds": reference["relative_seconds"],
            "gold_state": reference["execution_state"],
            "predicted_state": parsed["state"] if parsed else "__invalid__",
        })
    target_state = "recovery" if args.event_kind == "recovery_onset" else "failure"
    result = evaluate_failure_timing(rows, consecutive=args.consecutive, target_state=target_state)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", "detection_delay_seconds", "false_alarms_before_onset", "windows"])
        writer.writeheader()
        for episode, values in result["episodes"].items():
            writer.writerow({
                "episode_index": episode,
                "detection_delay_seconds": values["detection_delay_seconds"],
                "false_alarms_before_onset": values["false_alarms_before_onset"],
                "windows": values["windows"],
            })
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
