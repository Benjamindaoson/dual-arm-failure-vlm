from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.gates import decide_stage


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence gate for REBOOT post-training stages")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "evidence_gates.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = _read(args.config)
    result = decide_stage(
        _read(args.base),
        _read(args.sft) if args.sft else None,
        target_failure_recall=float(config["base"]["failure_recall"]),
        target_state_macro_f1=float(config["base"]["state_macro_f1"]),
        min_failure_gain=float(config["sft_minimum_gain"]["failure_recall"]),
        min_state_gain=float(config["sft_minimum_gain"]["state_macro_f1"]),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
