from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence gate for REBOOT post-training stages")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, default=None)
    parser.add_argument("--target-failure-recall", type=float, default=0.90)
    parser.add_argument("--target-state-macro-f1", type=float, default=0.85)
    parser.add_argument("--min-rl-headroom", type=float, default=0.03)
    args = parser.parse_args()

    base = _read(args.base)
    if args.sft is None:
        needs_sft = (
            float(base.get("failure_recall", 0.0)) < args.target_failure_recall
            or float(base.get("state_macro_f1", 0.0)) < args.target_state_macro_f1
        )
        result = {
            "decision": "RUN_SFT" if needs_sft else "STOP_BASE_SUFFICIENT",
            "reason": "held-out base critic leaves measurable headroom" if needs_sft else "base meets frozen pilot targets",
        }
    else:
        sft = _read(args.sft)
        base_score = float(base.get("state_macro_f1", 0.0))
        sft_score = float(sft.get("state_macro_f1", 0.0))
        sft_failure = float(sft.get("failure_recall", 0.0))
        needs_rl = (
            sft_failure < args.target_failure_recall
            or sft_score < args.target_state_macro_f1
        )
        learned = sft_score >= base_score + args.min_rl_headroom
        result = {
            "decision": "RUN_RLVR" if needs_rl and learned else ("STOP_SFT_SUFFICIENT" if not needs_rl else "REVISIT_DATA_OR_SFT"),
            "base_state_macro_f1": base_score,
            "sft_state_macro_f1": sft_score,
            "sft_failure_recall": sft_failure,
            "reason": (
                "SFT helps but leaves outcome-level errors" if needs_rl and learned
                else ("SFT meets frozen pilot targets" if not needs_rl else "SFT did not establish a clean learning signal")
            ),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
