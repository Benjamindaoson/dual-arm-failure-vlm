from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.gates import decide_v2_rl_gate
from reboot_recovery.metrics import diagnose_predictions
from reboot_recovery.rewards import score_state_gated_prediction


def _read(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {str(row["id"]): row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise SystemExit(f"empty or duplicate prediction IDs: {path}")
    return indexed


def _verify_reward(references: list[dict]) -> bool:
    modes = [str(ref["failure_mode"]) for ref in references if ref["failure_mode"] != "none"]
    fallback_mode = modes[0] if modes else "misalignment"
    for ref in references:
        exact = {"phase": ref["phase_name"], "state": ref["execution_state"], "failure_mode": ref["failure_mode"]}
        if score_state_gated_prediction(ref, json.dumps(exact)) != 1.0:
            return False
        for wrong_state in {"nominal", "failure", "recovery"} - {str(ref["execution_state"])}:
            wrong = {**exact, "state": wrong_state,
                     "failure_mode": "none" if wrong_state == "nominal" else (ref["failure_mode"] if ref["failure_mode"] != "none" else fallback_mode)}
            if score_state_gated_prediction(ref, json.dumps(wrong)) != 0.0:
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Failure-first V2 RL gate on paired full-schema validation predictions")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--sft-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.base_predictions.resolve(), args.sft_predictions.resolve()}:
        raise SystemExit("gate output must not overwrite raw predictions")
    base, sft = _read(args.base_predictions), _read(args.sft_predictions)
    if base.keys() != sft.keys():
        raise SystemExit("paired validation IDs differ")
    hashes = {row.get("split_receipt_sha256") for row in [*base.values(), *sft.values()]}
    if len(hashes) != 1 or None in hashes:
        raise SystemExit("split receipt hash differs or is missing")
    for sample_id in base:
        reference = base[sample_id]["reference"]
        if reference != sft[sample_id]["reference"] or reference.get("split") != "val":
            raise SystemExit(f"non-matching or non-validation reference: {sample_id}")
    base_metrics = diagnose_predictions(base.values())["semantic"]
    sft_metrics = diagnose_predictions(sft.values())["semantic"]
    verifier_validated = _verify_reward([row["reference"] for row in base.values()])
    decision = decide_v2_rl_gate(
        {**base_metrics, "task_schema": "full"}, {**sft_metrics, "task_schema": "full"},
        verifier_validated=verifier_validated,
    )
    decision.update({
        "status": "VALIDATION_GATE", "split_receipt_sha256": next(iter(hashes)),
        "sample_ids": sorted(base),
        "base_predictions_sha256": file_sha256(args.base_predictions),
        "sft_predictions_sha256": file_sha256(args.sft_predictions),
        "base_semantic_metrics": base_metrics, "sft_semantic_metrics": sft_metrics,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, decision)
    print(json.dumps({key: decision[key] for key in ("decision", "failure_correct_gain", "verifier_validated")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
