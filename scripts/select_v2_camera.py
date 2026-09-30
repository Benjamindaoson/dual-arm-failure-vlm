from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_camera(metrics: dict[str, dict]) -> str:
    """Select using validation failure recall, then state F1, then fixed tie order."""
    order = ("C0", "C1", "C2")
    if set(metrics) != set(order):
        raise ValueError("C0/C1/C2 validation metrics are all required")
    return max(order, key=lambda name: (
        float(metrics[name]["failure_recall"]),
        float(metrics[name]["state_macro_f1"]),
        -order.index(name),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = {}
    for name in ("C0", "C1", "C2"):
        path = args.runs_root / f"{name.lower()}-sft-val" / "metrics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        metrics[name] = value["semantic_diagnostic"]
    selected = select_camera(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "selection_set": "validation_only", "criterion": "failure_recall_then_state_macro_f1_then_C0_C1_C2",
        "selected": selected, "metrics": metrics,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
