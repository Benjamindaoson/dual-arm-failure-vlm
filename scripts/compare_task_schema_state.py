from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_compare import compare

from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.metrics import parse_state_prediction
from reboot_recovery.rewards import parse_structured_prediction
from reboot_recovery.semantic_parser import normalize_semantic_output


def project_state(rows: list[dict], schema: str) -> list[dict]:
    parser = parse_state_prediction if schema == "state-only" else parse_structured_prediction
    projected = []
    for row in rows:
        text = normalize_semantic_output(str(row.get("output", "")))
        parsed = parser(text)
        projected.append({**row, "output": json.dumps({"state": parsed["state"]}) if parsed else text})
    return projected


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired semantic state comparison across output schemas")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--first-schema", choices=["state-only", "full"], required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--second-schema", choices=["state-only", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.first.resolve(), args.second.resolve()}:
        raise SystemExit("comparison output cannot overwrite predictions")
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = compare(project_state(read(args.first), args.first_schema),
                     project_state(read(args.second), args.second_schema),
                     task_schema="state-only", semantic=False)
    result.update({"protocol": "semantic_state_projection", "first_schema": args.first_schema,
                   "second_schema": args.second_schema,
                   "source_sha256": {"first": file_sha256(args.first), "second": file_sha256(args.second)}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result["second_minus_first"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
