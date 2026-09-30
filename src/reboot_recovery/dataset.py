from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_local_dataset(root: str | Path, *, require_full: bool = False) -> dict[str, Any]:
    base = Path(root)
    required = [Path("meta/info.json"), Path("meta/phase.json")]
    if require_full:
        required.extend((Path("data"), Path("videos")))
    missing = [path.as_posix() for path in required if not (base / path).exists()]
    payload_files = {
        name: sum(1 for path in (base / name).rglob("*") if path.is_file()) if (base / name).is_dir() else 0
        for name in ("data", "videos")
    }
    if require_full:
        missing.extend(f"{name}/*" for name, count in payload_files.items() if count == 0)
    parse_errors: list[str] = []
    for relative in (Path("meta/info.json"), Path("meta/phase.json")):
        path = base / relative
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                parse_errors.append(f"{relative.as_posix()}:{error}")
    return {
        "root": str(base.resolve()),
        "valid": not missing and not parse_errors,
        "metadata_present": all((base / path).is_file() for path in required[:2]),
        "full_payload_present": all(payload_files.values()),
        "payload_file_counts": payload_files,
        "missing": missing,
        "parse_errors": parse_errors,
    }
