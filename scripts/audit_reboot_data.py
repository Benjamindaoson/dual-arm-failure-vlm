from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.annotations import load_annotations, split_episode_ids, usable_episode
from reboot_recovery.audit import audit_dataset, audit_frame_payload
from reboot_recovery.splits import build_split_receipt, canonical_sha256


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an immutable REBOOT metadata snapshot and create episode-safe split receipts")
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True, help="Hugging Face dataset API JSON containing id and sha")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "data_audit")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "artifacts" / "splits")
    parser.add_argument("--source-endpoint", default="local_snapshot")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Full local snapshot for frame-table validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    info, phase, repository = _read(args.info), _read(args.phase), _read(args.repository)
    result = audit_dataset(info, phase, repository)
    if args.dataset_root is not None:
        frame_audit = audit_frame_payload(str(args.dataset_root), phase)
        result.annotation_audit["frame_table_audit"] = frame_audit
        issues = dict(result.annotation_audit["issues_by_episode"])
        for episode_id in frame_audit["count_mismatches"]:
            issues.setdefault(episode_id, []).append("frame_table_count_mismatch")
        for row in frame_audit["invalid_frame_rows"]:
            episode_id = row.get("episode_index")
            if episode_id is not None:
                issues.setdefault(str(episode_id), []).append("invalid_frame_index")
        issues = {key: sorted(set(values)) for key, values in issues.items()}
        quarantined = sorted(issues)
        observed_ids = {
            str(row.get("episode_index")) for row in phase.get("episodes", []) if isinstance(row, dict)
        }
        result.annotation_audit.update({
            "issues_by_episode": issues,
            "issue_counts": dict(sorted(Counter(issue for values in issues.values() for issue in values).items())),
            "quarantined_episode_ids": quarantined,
            "valid_episode_ids": sorted(observed_ids - set(quarantined)),
            "episodes_quarantined": len(quarantined),
            "episodes_valid": len(observed_ids - set(quarantined)),
        })
        valid_ids = set(result.annotation_audit["valid_episode_ids"])
        valid_rows = [
            row for row in phase.get("episodes", [])
            if isinstance(row, dict) and str(row.get("episode_index")) in valid_ids
        ]
        result.failure_mode_distribution["valid_episodes"] = dict(sorted(Counter(
            str(row["failure"]["failure_mode"])
            for row in valid_rows if isinstance(row.get("failure"), dict)
        ).items()))
        phase_names = phase.get("phase_names", [])
        result.phase_distribution["valid_episodes"] = dict(sorted(Counter(
            str(phase_names[int(row["failure"]["originating_phase"]) - 1])
            for row in valid_rows if isinstance(row.get("failure"), dict)
        ).items()))
    source_hashes = {path.name: _sha256(path) for path in (args.info, args.phase, args.repository)}
    schema = dict(result.schema)
    schema["source_endpoint"] = args.source_endpoint
    schema["source_sha256"] = source_hashes
    _write(args.output_dir / "reboot_schema.json", schema)
    _write(args.output_dir / "reboot_annotation_audit.json", result.annotation_audit)
    _write(args.output_dir / "failure_mode_distribution.json", result.failure_mode_distribution)
    _write(args.output_dir / "phase_distribution.json", result.phase_distribution)

    annotations = load_annotations(args.phase)
    audited_valid_ids = set(result.annotation_audit["valid_episode_ids"])
    usable = [
        episode for episode in annotations.episodes
        if usable_episode(episode) and episode.episode_index in audited_valid_ids
    ]
    assignments = split_episode_ids(usable, seed=args.seed)
    split_receipt = build_split_receipt(usable, assignments, seed=args.seed)
    manifest = {
        "dataset_repo": repository.get("id"),
        "dataset_revision": repository.get("sha"),
        "episode_to_split": dict(sorted(assignments.items())),
        "episodes": split_receipt["episodes"],
        "seed": args.seed,
        "unit": "episode",
    }
    manifest["sha256"] = canonical_sha256(manifest)
    _write(args.split_dir / "split_manifest.json", manifest)
    receipt = dict(split_receipt)
    receipt.update({
        "dataset_repo": repository.get("id"),
        "dataset_revision": repository.get("sha"),
        "source_annotation_sha256": source_hashes[args.phase.name],
        "split_manifest_sha256": manifest["sha256"],
        "quarantined_episode_ids": result.annotation_audit["quarantined_episode_ids"],
    })
    _write(args.split_dir / "split_receipt.json", receipt)

    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# REBOOT Pilot Data Receipt",
        "",
        f"- Dataset: `{repository.get('id')}`",
        f"- Immutable revision: `{repository.get('sha')}`",
        f"- Last modified: `{repository.get('lastModified')}`",
        f"- Retrieved through: `{args.source_endpoint}`",
        f"- Receipt generated (UTC): `{generated}`",
        f"- Episodes observed: **{result.annotation_audit['episodes_observed']}**",
        f"- Episodes usable: **{result.annotation_audit['episodes_valid']}**",
        f"- Episodes quarantined: **{result.annotation_audit['episodes_quarantined']}**",
        f"- Frames declared by `meta/info.json`: **{info.get('total_frames')}**",
        f"- Sum of annotation terminal frame indices: **{result.annotation_audit['annotated_duration_frames']}**",
        f"- Expected rows under inclusive terminal-index semantics: **{result.annotation_audit['expected_rows_from_inclusive_duration']}**",
        f"- FPS: **{info.get('fps')}**",
        f"- RGB cameras: `{', '.join(result.schema['rgb_camera_keys'])}`",
        f"- Depth present in this pilot snapshot: **{result.schema['has_depth']}**",
        f"- Robot state/action shape: `{result.schema['robot_state']['shape']}` / `{result.schema['action']['shape']}`",
        "",
        "## Integrity notes",
        "",
        "- `episode_index` is `int64` in the frame-table schema but encoded as a zero-padded string in `meta/phase.json`; consumers normalize it explicitly.",
        "- Official REBOOT phase numbering is one-based (1–5). Out-of-range values are quarantined rather than remapped.",
        "- The annotation-duration sum does not equal `total_frames`; the mismatch is retained as evidence and is not silently repaired.",
        (
            "- Full frame tables were audited against the annotations; RGB video decoding and model execution are separate evidence."
            if args.dataset_root is not None else
            "- This receipt audits metadata only. It does not claim that RGB frames, model weights, or GPU experiments were executed."
        ),
        "",
        "## Source SHA-256",
        "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(source_hashes.items()))
    (args.output_dir / "DATA_RECEIPT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset_revision": repository.get("sha"),
        "episodes_valid": result.annotation_audit["episodes_valid"],
        "episodes_quarantined": result.annotation_audit["episodes_quarantined"],
        "split_manifest_sha256": manifest["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
