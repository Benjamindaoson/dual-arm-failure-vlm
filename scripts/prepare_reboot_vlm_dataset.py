from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.ablations import resolve_ablation, select_window_frames, summarize_trace
from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.prompts import build_prompt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _target(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "phase": row["phase_name"],
            "state": row["execution_state"],
            "failure_mode": row["failure_mode"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _window_prompt(
    row: dict[str, Any], trace_summary: str | None = None, failure_modes: tuple[str, ...] = ()
) -> str:
    class _Obj:
        task_description = row["task_description"]
    return build_prompt(
        _Obj(), include_trace_summary=trace_summary is not None, trace_summary=trace_summary,
        failure_modes=failure_modes,
    )  # type: ignore[arg-type]


def _tensor_to_pil(value: Any):
    from PIL import Image
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.ndim == 3 and value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        value = value.numpy()
    value = np.asarray(value)
    if value.dtype != np.uint8:
        if value.max(initial=0) <= 1.0:
            value = value * 255.0
        value = value.clip(0, 255).astype("uint8")
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return Image.fromarray(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize REBOOT temporal windows as a standard VLM JSONL+image dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", default="REBOOT26/sample_recovery-demonstration")
    parser.add_argument("--root", type=Path, default=None, help="Optional local LeRobot dataset root")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared" / "reboot_vlm")
    parser.add_argument("--cameras", nargs="+", default=["observation.images.cam_high", "observation.images.cam_low"])
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--ablation", choices=["A0", "A1", "A2", "A3"], default="A2")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--split-receipt", type=Path, default=ROOT / "artifacts" / "splits" / "split_receipt.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.manifest)
    if not args.split_receipt.is_file():
        raise SystemExit(f"split receipt not found: {args.split_receipt}")
    split_receipt_sha256 = file_sha256(args.split_receipt)
    failure_modes = tuple(sorted({str(row["failure_mode"]) for row in rows if row["failure_mode"] != "none"}))
    spec = resolve_ablation(args.ablation, num_frames=args.num_frames, cameras=args.cameras)
    requested = {(int(r["episode_index"]), int(f)) for r in rows for f in select_window_frames(r, spec)}
    if args.dry_run:
        print(json.dumps({
            "windows": len(rows),
            "unique_episode_frame_pairs": len(requested),
            "ablation": spec.name,
            "num_frames": spec.num_frames,
            "cameras": spec.cameras,
            "include_trace": spec.include_trace,
            "estimated_images": len(requested) * len(spec.cameras),
            "split_receipt_sha256": split_receipt_sha256,
        }, indent=2))
        return 0

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit("Install LeRobot >=0.4.3 before materializing frames") from exc

    kwargs: dict[str, Any] = {"repo_id": args.repo_id, "video_backend": args.video_backend}
    if args.root is not None:
        kwargs["root"] = args.root
    dataset = LeRobotDataset(**kwargs)
    index_table = dataset.hf_dataset.select_columns(["episode_index", "frame_index"])
    lookup: dict[tuple[int, int], int] = {}
    for row_idx, meta in enumerate(index_table):
        key = (int(meta["episode_index"]), int(meta["frame_index"]))
        if key in requested:
            lookup[key] = row_idx
    missing = requested - lookup.keys()
    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(f"missing {len(missing)} requested episode/frame pairs; first={preview}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.output_dir / "images"
    image_root.mkdir(exist_ok=True)
    out_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for window_idx, row in enumerate(rows):
        images: list[str] = []
        states: list[Any] = []
        actions: list[Any] = []
        for frame in select_window_frames(row, spec):
            item = dataset[lookup[(int(row["episode_index"]), int(frame))]]
            if spec.include_trace:
                states.append(item["observation.state"])
                actions.append(item["action"])
            for camera in spec.cameras:
                if camera not in item:
                    raise KeyError(f"camera {camera!r} missing from dataset item")
                pil = _tensor_to_pil(item[camera])
                safe_cam = camera.rsplit(".", 1)[-1]
                rel = Path("images") / f"w{window_idx:04d}_f{int(frame):04d}_{safe_cam}.jpg"
                pil.convert("RGB").save(args.output_dir / rel, quality=90)
                images.append(str(rel))

        trace_summary = summarize_trace(states, actions) if spec.include_trace else None
        image_blocks = [{"type": "image"} for _ in images]
        anchor_kind = row.get("anchor_kind", row.get("event_kind", "window"))
        record = {
            "id": row.get("id", f"ep{row['episode_index']}-{anchor_kind}-f{row['anchor_frame']}"),
            "images": images,
            "ablation": spec.name,
            "trace_summary": trace_summary,
            "prompt": [{"role": "user", "content": image_blocks + [{"type": "text", "text": _window_prompt(row, trace_summary, failure_modes)}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": _target(row)}]}],
            "reference": row,
            "split_receipt_sha256": split_receipt_sha256,
        }
        out_by_split[row["split"]].append(record)

    for split, split_rows in out_by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in split_rows:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(split_rows)} -> {path}")
    write_json(args.output_dir / "dataset_receipt.json", {
        "manifest": str(args.manifest.resolve()),
        "split_receipt": str(args.split_receipt.resolve()),
        "split_receipt_sha256": split_receipt_sha256,
        "ablation": spec.name,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
