from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.evidence import environment_snapshot, utc_now, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the CUDA environment and fail closed when no GPU is visible")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "environment" / "gpu_environment.json")
    args = parser.parse_args(argv)
    snapshot = environment_snapshot()
    runtime = snapshot["torch_runtime"]
    payload = {
        "status": "GPU_READY" if runtime["cuda_available"] else "BLOCKED_NO_CUDA",
        "captured_at": utc_now(),
        **snapshot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if runtime["cuda_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
