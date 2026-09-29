from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(output_dir: str | Path, resume: str | None) -> Path | None:
    root = Path(output_dir)
    if not resume:
        return None
    if resume == "latest":
        candidates = []
        for path in root.glob("checkpoint-*"):
            if path.is_dir():
                try:
                    candidates.append((int(path.name.rsplit("-", 1)[-1]), path))
                except ValueError:
                    continue
        return max(candidates)[1] if candidates else None
    path = Path(resume)
    if not path.is_absolute():
        path = root / path
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path
