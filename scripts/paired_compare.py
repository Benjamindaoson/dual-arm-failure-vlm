from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reboot_recovery.evidence import file_sha256, write_json
from reboot_recovery.metrics import evaluate_predictions, evaluate_state_predictions, parse_state_prediction
from reboot_recovery.rewards import parse_structured_prediction
from reboot_recovery.semantic_parser import normalize_semantic_output


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = p * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def compare(first: list[dict], second: list[dict], *, task_schema: str,
            semantic: bool = True, samples: int = 1000, seed: int = 42) -> dict:
    if task_schema not in {"full", "state-only"} or samples < 1:
        raise ValueError("invalid task schema or bootstrap sample count")
    a = {row["id"]: row for row in first}
    b = {row["id"]: row for row in second}
    if not a or len(a) != len(first) or len(b) != len(second) or a.keys() != b.keys():
        raise ValueError("paired prediction IDs must match and be unique")
    groups: dict[str, list[str]] = defaultdict(list)
    for key in a:
        if a[key]["reference"] != b[key]["reference"] or a[key].get("split_receipt_sha256") != b[key].get("split_receipt_sha256"):
            raise ValueError(f"reference or split receipt differs: {key}")
        groups[str(a[key]["reference"]["episode_index"])].append(key)
    episode_ids = sorted(groups)
    metric = evaluate_state_predictions if task_schema == "state-only" else evaluate_predictions

    def score(indexed: dict, keys: list[str]) -> dict:
        rows = [indexed[key] for key in keys]
        if semantic:
            rows = [{**row, "output": normalize_semantic_output(str(row.get("output", "")))} for row in rows]
        return metric(rows, bootstrap_samples=0)

    keys = sorted(a)
    base, candidate = score(a, keys), score(b, keys)
    parser = parse_state_prediction if task_schema == "state-only" else parse_structured_prediction
    outcomes = {state: {name: 0 for name in ("both_correct", "rescued", "regressed", "both_wrong")}
                for state in ("nominal", "failure", "recovery")}
    for key in keys:
        gold = str(a[key]["reference"]["execution_state"])
        first_text, second_text = (str(a[key].get("output", "")), str(b[key].get("output", "")))
        if semantic:
            first_text, second_text = normalize_semantic_output(first_text), normalize_semantic_output(second_text)
        first_state = (parser(first_text) or {}).get("state")
        second_state = (parser(second_text) or {}).get("state")
        label = ("both_correct" if first_state == second_state == gold else
                 "rescued" if first_state != gold and second_state == gold else
                 "regressed" if first_state == gold and second_state != gold else "both_wrong")
        outcomes[gold][label] += 1
    # This rate is measured only on nominal windows; it is not onset-relative.
    endpoints = ("failure_recall", "state_macro_f1", "nominal_false_alarm_rate")
    source_endpoint = {"nominal_false_alarm_rate": "pre_failure_false_positive_rate"}
    value = lambda metrics, name: float(metrics[source_endpoint.get(name, name)])
    differences = {name: [] for name in endpoints}
    rng = random.Random(seed)
    for _ in range(samples):
        chosen = [rng.choice(episode_ids) for _ in episode_ids]
        resampled_keys = [key for episode in chosen for key in groups[episode]]
        am, bm = score(a, resampled_keys), score(b, resampled_keys)
        for name in endpoints:
            differences[name].append(value(bm, name) - value(am, name))
    return {
        "protocol": "semantic" if semantic else "strict", "task_schema": task_schema,
        "episode_support": len(episode_ids), "sample_support": len(keys),
        "class_support": base["gold_state_counts"], "bootstrap_samples": samples, "seed": seed,
        "first": {name: value(base, name) for name in endpoints},
        "second": {name: value(candidate, name) for name in endpoints},
        "paired_state_outcomes": outcomes,
        "second_minus_first": {name: {"point": value(candidate, name) - value(base, name),
                                      "ci95": [_percentile(differences[name], 0.025),
                                               _percentile(differences[name], 0.975)]}
                               for name in endpoints},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--task-schema", choices=["full", "state-only"], required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.first.resolve(), args.second.resolve()}:
        raise SystemExit("comparison output cannot overwrite predictions")
    rows = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = compare(rows(args.first), rows(args.second), task_schema=args.task_schema,
                     semantic=not args.strict, samples=args.samples, seed=args.seed)
    result["source_sha256"] = {"first": file_sha256(args.first), "second": file_sha256(args.second)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result["second_minus_first"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
