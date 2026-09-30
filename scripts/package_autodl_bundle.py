from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_FILES = {"README.md", "pyproject.toml", "requirements-reboot.txt"}
ALLOWED_DIRECTORIES = {"src", "scripts", "tests", "configs", "artifacts", "docs", "openspec", "upgraded_implementation"}
ALLOWED_SUFFIXES = {".py", ".sh", ".json", ".jsonl", ".md", ".txt", ".toml", ".csv", ".html"}
SENSITIVE_MARKERS = (".env", "credential", "secret", "id_rsa", "id_ed25519")


def bundle_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True, capture_output=True,
    )
    candidates = [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]
    files = []
    for relative in candidates:
        lowered = relative.as_posix().casefold()
        if any(marker in lowered for marker in SENSITIVE_MARKERS):
            continue
        allowed_location = relative.as_posix() in ALLOWED_ROOT_FILES or relative.parts[0] in ALLOWED_DIRECTORIES
        path = root / relative
        if allowed_location and path.is_file() and path.suffix.casefold() in ALLOWED_SUFFIXES and path.stat().st_size <= 5 * 1024 * 1024:
            files.append(path)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic source-and-receipt bundle for offline AutoDL use")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "reboot-autodl-bundle.zip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    files = bundle_files(ROOT)
    summary = {"files": len(files), "bytes": sum(path.stat().st_size for path in files), "output": str(args.output)}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix == ".sh" else 0o644) << 16
            archive.writestr(info, path.read_bytes())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
