from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def check_run(path: Path, steps: int) -> dict[str, object]:
    receipt = json.loads((path / "run_receipt.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    log = json.loads((path / "train_log.json").read_text(encoding="utf-8"))["log_history"]
    updates = [entry for entry in log if all(key in entry for key in ("loss", "grad_norm", "reward", "kl"))]
    if receipt["status"] != "COMPLETED" or metrics["global_step"] != steps or len(updates) < steps:
        raise ValueError("RL run incomplete or missing per-step diagnostics")
    for entry in updates:
        if not all(math.isfinite(float(entry[key])) for key in ("loss", "grad_norm", "reward", "kl")):
            raise ValueError(f"nonfinite RL optimization diagnostic at step {entry.get('step')}")
    adapter = path / "adapter_model.safetensors"
    if not adapter.is_file() or adapter.stat().st_size == 0:
        raise ValueError("RL adapter was not saved")
    return {"status": "STABLE", "steps": steps, "adapter_bytes": adapter.stat().st_size,
            "final_reward": updates[-1]["reward"], "final_kl": updates[-1]["kl"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    result = check_run(args.run, args.steps)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
