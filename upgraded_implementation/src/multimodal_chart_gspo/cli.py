from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chart reasoning outputs stored as JSONL.")
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    with args.records.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    print(json.dumps(evaluate_records(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
