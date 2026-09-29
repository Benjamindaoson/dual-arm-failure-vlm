from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download REBOOT pilot metadata or the full 2.84 GB pilot dataset")
    parser.add_argument("--repo-id", default="REBOOT26/sample_recovery-demonstration")
    parser.add_argument("--local-dir", type=Path, default=Path("data/reboot_sample"))
    parser.add_argument("--full", action="store_true", help="Download data/videos as well as metadata")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    allow_patterns = None if args.full else ["meta/*", "README.md"]
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
