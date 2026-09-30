from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.metrics import diagnose_predictions


def _source_receipt_evidence(path: Path, row_count: int) -> dict[str, str]:
    receipt_path = path.parent / "run_receipt.json"
    if not receipt_path.is_file():
        return {"source_receipt_verification": "NO_RECEIPT"}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt["output_sha256"]["predictions.jsonl"]
    if file_sha256(path) == expected:
        status = "BYTE_MATCH"
    else:
        raw = path.read_bytes()
        canonical = raw.replace(b"\r\n", b"\n")
        if raw.count(b"\r\n") != row_count or raw.count(b"\r") != row_count or hashlib.sha256(canonical).hexdigest() != expected:
            raise SystemExit(f"prediction bytes do not match source run receipt: {path}")
        status = "CRLF_CHECKOUT_MATCH"
    return {"source_receipt_verification": status, "source_receipt_sha256": expected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate strict protocol and mechanical semantic scores")
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "PREDICTIONS"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    results = {}
    baseline = None
    split_hash = None
    input_paths = set()
    for name, raw_path in args.run:
        path = Path(raw_path)
        if name in results:
            raise SystemExit(f"duplicate run name: {name}")
        if not path.is_file():
            raise SystemExit(f"prediction file not found: {path}")
        input_paths.add(path.resolve())
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"empty prediction file: {path}")
        indexed = {str(row["id"]): row for row in rows}
        if len(indexed) != len(rows):
            raise SystemExit(f"duplicate sample IDs: {path}")
        identity = {key: (row["reference"], row.get("split_receipt_sha256")) for key, row in indexed.items()}
        hashes = {row.get("split_receipt_sha256") for row in rows}
        if len(hashes) != 1 or None in hashes:
            raise SystemExit(f"missing or inconsistent split receipt: {path}")
        if baseline is not None and identity != baseline:
            raise SystemExit(f"paired sample references differ: {path}")
        baseline = identity
        split_hash = hashes.pop()
        results[name] = {"input_sha256": file_sha256(path), **_source_receipt_evidence(path, len(rows)), **diagnose_predictions(rows)}
    if args.output.resolve() in input_paths:
        raise SystemExit("output must not overwrite raw predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "status": "RETROSPECTIVE_DIAGNOSTIC",
        "normalization": "outer whitespace and one exact lowercase json fence only",
        "split_receipt_sha256": split_hash,
        "sample_ids": sorted(baseline),
        "runs": results,
    })
    print(json.dumps({name: {"n": value["protocol"]["n"], "semantic_failure_recall": value["semantic"]["failure_recall"]}
                      for name, value in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
