from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.metrics import evaluate_predictions
from reboot_recovery.rewards import score_prediction


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate prediction IDs in {path}")
    return indexed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired comparison of two model prediction files")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="Base")
    parser.add_argument("--right-name", default="SFT")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--error-cases", type=Path, default=None)
    args = parser.parse_args(argv)

    left, right = read_predictions(args.left), read_predictions(args.right)
    if set(left) != set(right):
        raise SystemExit("paired prediction IDs differ")
    hashes = {
        str(row.get("split_receipt_sha256"))
        for row in [*left.values(), *right.values()]
    }
    if len(hashes) != 1 or "None" in hashes:
        raise SystemExit(f"split receipt hash is missing or inconsistent: {sorted(hashes)}")

    categories = {"left_wrong_right_correct": 0, "left_correct_right_wrong": 0, "both_correct": 0, "both_wrong": 0}
    error_rows = []
    for sample_id in sorted(left):
        left_row, right_row = left[sample_id], right[sample_id]
        left_correct = score_prediction(left_row["reference"], str(left_row.get("output", ""))).total == 1.0
        right_correct = score_prediction(right_row["reference"], str(right_row.get("output", ""))).total == 1.0
        category = (
            "both_correct" if left_correct and right_correct else
            "left_wrong_right_correct" if right_correct else
            "left_correct_right_wrong" if left_correct else
            "both_wrong"
        )
        categories[category] += 1
        if category != "both_correct":
            error_rows.append({
                "id": sample_id, "category": category, "reference": left_row["reference"],
                args.left_name: left_row.get("output"), args.right_name: right_row.get("output"),
            })

    metrics = {
        args.left_name: evaluate_predictions(left.values()),
        args.right_name: evaluate_predictions(right.values()),
    }
    payload = {
        "status": "EXECUTED", "split_receipt_sha256": hashes.pop(),
        "samples": len(left), "paired_outcomes": categories, "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metric_names = ["state_macro_f1", "failure_recall", "recovery_recall", "phase_macro_f1", "failure_mode_macro_f1", "json_valid_rate", "pre_failure_false_positive_rate"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *metric_names])
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"model": name, **{metric: values[metric] for metric in metric_names}})
    if args.error_cases:
        args.error_cases.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in error_rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
