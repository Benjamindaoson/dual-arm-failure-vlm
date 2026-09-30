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


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare A0-A3 on identical held-out REBOOT episodes")
    for name in ("a0", "a1", "a2", "a3"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "eval" / "input_ablation.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "artifacts" / "eval" / "input_ablation.csv")
    args = parser.parse_args(argv)
    runs = {name.upper(): _read(getattr(args, name)) for name in ("a0", "a1", "a2", "a3")}
    episode_sets = {name: sorted({str(row["reference"]["episode_index"]) for row in rows}) for name, rows in runs.items()}
    sample_ids = {name: sorted(str(row["id"]) for row in rows) for name, rows in runs.items()}
    receipt_hashes = {
        name: sorted({str(row.get("split_receipt_sha256")) for row in rows}) for name, rows in runs.items()
    }
    if len({tuple(value) for value in episode_sets.values()}) != 1:
        raise SystemExit(f"ablation test episodes differ: {episode_sets}")
    if len({tuple(value) for value in sample_ids.values()}) != 1:
        raise SystemExit("ablation sample IDs or counts differ")
    if any(values == ["None"] or len(values) != 1 for values in receipt_hashes.values()) or len({tuple(value) for value in receipt_hashes.values()}) != 1:
        raise SystemExit(f"ablation split receipts differ or are missing: {receipt_hashes}")
    results = {name: evaluate_predictions(rows) for name, rows in runs.items()}
    payload = {
        "status": "EXECUTED",
        "split_receipt_sha256": next(iter(receipt_hashes.values()))[0],
        "test_episode_ids": next(iter(episode_sets.values())),
        "sample_ids": next(iter(sample_ids.values())),
        "variants": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["variant", "n", "state_macro_f1", "failure_recall", "recovery_recall", "phase_macro_f1", "failure_mode_macro_f1", "json_valid_rate", "pre_failure_false_positive_rate"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"variant": name, **{field: metrics[field] for field in fields[1:]}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
