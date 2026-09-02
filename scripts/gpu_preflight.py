from __future__ import annotations
import argparse, json, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gpu_gspo.json"

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = {"base_model", "dataset_id", "dataset_split", "output_dir", "num_generations", "rewards"}
    missing = required - config.keys()
    if missing: raise SystemExit(f"missing config keys: {sorted(missing)}")
    print(json.dumps({"config": str(CONFIG), "run_name": config["run_name"], "dry_run": args.dry_run}, ensure_ascii=False))
    if args.dry_run: return
    if not shutil.which("nvidia-smi"): raise SystemExit("CUDA GPU not found; run this only on the GPU instance")
    subprocess.run(["nvidia-smi"], check=True)
if __name__ == "__main__": main()
