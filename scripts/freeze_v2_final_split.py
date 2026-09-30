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


def assign_fallback(usable_ids: set[str], development_split: dict[str, str], *, seed: int) -> dict[str, str]:
    if not set(development_split) <= usable_ids or {key for key, value in development_split.items() if value == "test"} != V1_DIAGNOSTIC_SET:
        raise ValueError("development split includes unusable episodes or differs from frozen V1 test")
    pool = sorted(key for key, value in development_split.items() if value == "train")
    if len(pool) < 18:
        raise ValueError("fallback requires at least 18 development-train episodes")
    random.Random(seed).shuffle(pool)
    return {episode: ("test" if index < 6 else "val" if index < 11 else "train")
            for index, episode in enumerate(pool)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a new, episode-disjoint final split after the V2 method commit")
    parser.add_argument("--phase-json", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--method-selection", type=Path, required=True)
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
    development = json.loads(args.development_split.read_text(encoding="utf-8"))
    if development["dataset_revision"] != args.dataset_revision:
        raise SystemExit("development split dataset revision differs")
    method = json.loads(args.method_selection.read_text(encoding="utf-8"))
    if method.get("selection_set") != "validation_only" or method.get("selected") not in {"sparse-visual", "dense-visual", "sparse-trace"}:
        raise SystemExit("method selection is missing or invalid")
    assignments = assign_fallback(usable, development["episode_to_split"], seed=args.seed)
    grouped = {name: sorted(key for key, value in assignments.items() if value == name)
               for name in ("train", "val", "test")}
    payload = {
        "tier": "B_SINGLE_TASK", "status": "FROZEN", "unit": "episode",
        "task_ids": [annotations.task_id], "dataset_revision": args.dataset_revision,
        "source_annotation_sha256": file_sha256(args.phase_json),
        "method_commit": args.method_commit, "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "uniform random over original development-train episodes only, six final and five validation",
        "limitation": "final-test episodes were used as training data during method development, so this is internal Tier B evidence, not fully independent replication",
        "seed": args.seed, "V1_DIAGNOSTIC_SET": sorted(V1_DIAGNOSTIC_SET),
        "excluded_development_val": sorted(key for key, value in development["episode_to_split"].items() if value == "val"),
        "development_split_sha256": file_sha256(args.development_split),
        "method_selection_sha256": file_sha256(args.method_selection),
        "selected_method": method["selected"], "selected_camera": method["camera"],
        "episode_to_split": assignments, "episodes": grouped,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(json.dumps({"episodes": grouped, "receipt_sha256": payload["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
