"""Snapshot the Qwen3-VL GSPO course notebook and its small result artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\备课\强化学习自学资料\【课件】大模型强化学习实战\阶段三：新兴大模型强化学习技术实战\视觉强化学习实战")
FILES = ["【视觉强化学习实战项目】Qwen3_VL_(8B)_GSPO.ipynb", "baseline_records.json", "after_records.json"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    copied = []
    for name in FILES:
        source, target = SOURCE / name, ROOT / "legacy_reproduction" / name
        if not source.exists():
            raise FileNotFoundError(source)
        copied.append({"source": str(source), "target": str(target), "sha256": digest(source), "bytes": source.stat().st_size})
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = {"source_root": str(SOURCE), "copied": copied, "external_only": [{"reason": "base model and visual training data are downloaded only on the GPU machine"}]}
    if not args.dry_run:
        (ROOT / "course_materials" / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
