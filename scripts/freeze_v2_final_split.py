from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reboot_recovery.annotations import load_annotations, usable_episode
from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.splits import canonical_sha256

V1_DIAGNOSTIC_SET = {"02", "08", "18", "20", "47", "54"}


def assign_fallback(usable_ids: set[str], *, seed: int) -> dict[str, str]:
    pool = sorted(usable_ids - V1_DIAGNOSTIC_SET)
    if len(pool) < 18 or not V1_DIAGNOSTIC_SET <= usable_ids:
        raise ValueError("fallback requires all six V1 diagnostic episodes and at least 18 other usable episodes")
    random.Random(seed).shuffle(pool)
    return {episode: ("test" if index < 6 else "val" if index < 11 else "train")
            for index, episode in enumerate(pool)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a new, episode-disjoint final split after the V2 method commit")
    parser.add_argument("--phase-json", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--method-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen split: {args.output}")
    commit = subprocess.run(["git", "cat-file", "-e", f"{args.method_commit}^{{commit}}"], cwd=ROOT)
    if commit.returncode:
        raise SystemExit("method commit must exist locally")
    annotations = load_annotations(args.phase_json)
    usable = {ep.episode_index for ep in annotations.episodes if usable_episode(ep)}
    assignments = assign_fallback(usable, seed=args.seed)
    grouped = {name: sorted(key for key, value in assignments.items() if value == name)
               for name in ("train", "val", "test")}
    payload = {
        "tier": "B_SINGLE_TASK", "status": "FROZEN", "unit": "episode",
        "task_ids": [annotations.task_id], "dataset_revision": args.dataset_revision,
        "source_annotation_sha256": file_sha256(args.phase_json),
        "method_commit": args.method_commit, "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "uniform random over usable non-V1-diagnostic episode IDs, six final and five validation",
        "seed": args.seed, "V1_DIAGNOSTIC_SET": sorted(V1_DIAGNOSTIC_SET),
        "episode_to_split": assignments, "episodes": grouped,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(json.dumps({"episodes": grouped, "receipt_sha256": payload["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
