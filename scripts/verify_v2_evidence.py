from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _hash_status(path: Path, expected: str) -> str:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return "BYTE_MATCH"
    if b"\r\n" in raw and hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
        return "CRLF_CHECKOUT_MATCH"
    raise ValueError(f"receipt hash mismatch: {path}")


def verify_run(directory: Path) -> dict:
    receipt = json.loads((directory / "run_receipt.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETED" or receipt["run_id"] != directory.name:
        raise ValueError(f"run is not completed or ID differs: {directory}")
    common = ("config.json", "environment.json", "metrics.json", "timing.json",
              "memory.json", "run_receipt.json", "stdout.log", "stderr.log")
    required = (*common, "train_log.json") if config["stage"] in {"sft", "rlvr"} else (*common, "predictions.jsonl")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"missing evidence files in {directory}: {missing}")
    hashes = {name: _hash_status(directory / name, value)
              for name, value in receipt["output_sha256"].items()
              if (directory / name).is_file()}
    external = {name: value for name, value in receipt["output_sha256"].items()
                if not (directory / name).is_file()}
    if any(not name.endswith(".safetensors") for name in external):
        raise ValueError(f"missing receipt output other than external weights: {directory}")
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if config_hash != receipt["config_sha256"]:
        raise ValueError(f"configuration hash mismatch: {directory}")
    return {"run_id": directory.name, "status": "VERIFIED", "hashed_outputs": hashes,
            "external_checkpoint_sha256": external, "git_commit": receipt.get("git_commit")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [verify_run(path) for path in sorted(args.runs_root.iterdir())
               if path.is_dir() and (path / "run_receipt.json").is_file()
               and json.loads((path / "run_receipt.json").read_text(encoding="utf-8")).get("status") == "COMPLETED"]
    if not results:
        raise SystemExit("no completed runs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"verified_run_count": len(results), "runs": results}, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(results)} completed runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
