from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.metrics import parse_state_prediction
from reboot_recovery.semantic_parser import normalize_semantic_output


def choose(metrics: dict[str, dict]) -> str:
    order = ("sparse-visual", "dense-visual", "sparse-trace")
    if set(metrics) != set(order):
        raise ValueError("all three matched validation conditions are required")
    return max(order, key=lambda name: (
        float(metrics[name]["failure_recall"]),
        float(metrics[name]["state_macro_f1"]),
        -order.index(name),
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--camera-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    camera = json.loads(args.camera_selection.read_text(encoding="utf-8"))["selected"]
    names = {
        "sparse-visual": f"{camera.lower()}-sft-val",
        "dense-visual": "dense-sft-val",
        "sparse-trace": "trace-sft-val",
    }
    metrics = {}
    prediction_hashes = {}
    detected_failure_episodes = {}
    paired = None
    for method, name in names.items():
        run = args.runs_root / name
        metrics[method] = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["semantic_diagnostic"]
        predictions = run / "predictions.jsonl"
        prediction_hashes[method] = file_sha256(predictions)
        rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        detected_failure_episodes[method] = sorted({str(row["reference"]["episode_index"]) for row in rows
            if row["reference"]["execution_state"] == "failure"
            and (parse_state_prediction(normalize_semantic_output(str(row.get("output", "")))) or {}).get("state") == "failure"})
        keys = {row["id"]: row["reference"] for row in rows}
        if not rows or len(keys) != len(rows) or (paired is not None and paired != keys):
            raise SystemExit(f"validation predictions are not paired: {name}")
        paired = keys
    selected = choose(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "selection_set": "validation_only", "camera": camera, "selected": selected,
        "criterion": "failure_recall_then_state_macro_f1_then_sparse_dense_trace",
        "metrics": metrics, "prediction_sha256": prediction_hashes,
        "detected_failure_episodes": detected_failure_episodes,
        "multitask_isolation_allowed": len(detected_failure_episodes[selected]) >= 2,
        "multitask_isolation_rule": "at_least_two_validation_episodes_with_correct_failure_prediction",
        "paired_sample_count": len(paired or {}),
    })
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
