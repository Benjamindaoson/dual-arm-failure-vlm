from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.dataset import verify_local_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a local REBOOT dataset root without network access")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = verify_local_dataset(args.dataset_root, require_full=args.require_full)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
