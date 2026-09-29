from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.rewards import score_prediction


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate structured REBOOT critic outputs")
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL rows with reference + output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [score_prediction(row["reference"], row["output"]) for row in rows]
    n = len(scores)
    metrics = {
        "n": n,
        "json_valid_rate": sum(x.valid_json for x in scores) / n if n else 0.0,
        "phase_accuracy": sum(x.phase for x in scores) / n if n else 0.0,
        "state_accuracy": sum(x.state for x in scores) / n if n else 0.0,
        "mean_task_reward": sum(x.total for x in scores) / n if n else 0.0,
    }
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
