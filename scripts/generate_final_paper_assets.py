"""Generate Tier-B submission assets only from saved, receipt-verified predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.metrics import evaluate_predictions, evaluate_state_predictions
from reboot_recovery.semantic_parser import normalize_semantic_output
from scripts.paired_compare import compare
from scripts.verify_v2_evidence import verify_run


RUNS = {
    "State Base": ("final-base-state", "state-only"),
    "State SFT 42": ("final-sft-state-seed42-test", "state-only"),
    "State SFT 43": ("final-sft-state-seed43-test", "state-only"),
    "State SFT 44": ("final-sft-state-seed44-test", "state-only"),
    "Full Base": ("final-base-full-test", "full"),
    "Full SFT": ("final-sft-full-seed42-test", "full"),
}
ANALYSIS_CODE = (
    "src/reboot_recovery/metrics.py",
    "src/reboot_recovery/semantic_parser.py",
    "src/reboot_recovery/rewards.py",
    "scripts/paired_compare.py",
    "scripts/generate_final_paper_assets.py",
    "scripts/verify_v2_evidence.py",
)
FROZEN_FINAL_SOURCES = (
    "artifacts/v2/manifests/final_sparse.jsonl",
    "artifacts/v2/manifests/final_sparse_dataset_receipt.json",
    "artifacts/v2/splits/paper_final_test_receipt.json",
)
PRIMARY_FIELDS = ("n", "json_valid_rate", "state_macro_f1", "failure_correct", "failure_support",
                  "failure_recall", "failure_precision", "pre_failure_false_positive_rate", "recovery_recall")


def text_sha256(path: Path) -> str:
    """Hash text independent of Git CRLF checkout translation."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows) or not rows:
        raise ValueError(f"empty or duplicate prediction IDs: {path}")
    return rows


def _prediction_counts(rows: list[dict], schema: str) -> dict[str, int]:
    from reboot_recovery.metrics import parse_state_prediction
    from reboot_recovery.rewards import parse_structured_prediction
    parser = parse_state_prediction if schema == "state-only" else parse_structured_prediction
    counts = {name: 0 for name in ("nominal", "failure", "recovery", "invalid")}
    for row in rows:
        parsed = parser(str(row["output"]))
        counts[parsed["state"] if parsed else "invalid"] += 1
    return counts


def _evaluate(rows: list[dict], schema: str, *, bootstrap_samples: int) -> dict:
    evaluator = evaluate_state_predictions if schema == "state-only" else evaluate_predictions
    return evaluator(rows, bootstrap_samples=bootstrap_samples)


def assert_same_final_references(runs: dict[str, list[dict]]) -> None:
    """Check the six conditions really share frozen IDs, labels, and split receipt."""
    reference: dict[str, tuple[dict, str | None]] | None = None
    for name, rows in runs.items():
        indexed = {str(row["id"]): (row["reference"], row.get("split_receipt_sha256")) for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate final sample ID: {name}")
        if reference is None:
            reference = indexed
        elif indexed != reference:
            raise ValueError(f"final sample IDs, references, or split receipt differ: {name}")


def assert_frozen_final_manifest(root: Path, rows: list[dict]) -> None:
    """Bind scored rows to the pre-inference test manifest and split receipt."""
    manifest = [json.loads(line) for line in (root / FROZEN_FINAL_SOURCES[0]).read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt = json.loads((root / FROZEN_FINAL_SOURCES[2]).read_text(encoding="utf-8"))
    materialization = json.loads((root / FROZEN_FINAL_SOURCES[1]).read_text(encoding="utf-8"))
    split_hash = text_sha256(root / FROZEN_FINAL_SOURCES[2])
    if materialization["split_receipt_sha256"] != split_hash or receipt.get("status") != "FROZEN":
        raise ValueError("final split receipt or materialization hash disagrees")
    test_manifest = [row for row in manifest if row["split"] == "test"]
    expected = {
        f"ep{row['episode_index']}-{row['anchor_kind']}-f{row['anchor_frame']}": row
        for row in test_manifest
    }
    if len(test_manifest) != 54 or len(expected) != 54 or set(expected_row["episode_index"] for expected_row in expected.values()) != set(receipt["episodes"]["test"]):
        raise ValueError("frozen final manifest does not match test receipt")
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows) or indexed.keys() != expected.keys():
        raise ValueError("prediction IDs differ from frozen final manifest")
    for sample_id, row in indexed.items():
        if row["reference"] != expected[sample_id] or row.get("split_receipt_sha256") != split_hash:
            raise ValueError(f"prediction reference differs from frozen manifest or receipt: {sample_id}")


def load_final_evidence(root: Path = ROOT, *, bootstrap_samples: int = 1000) -> dict:
    root = root.resolve()
    result: dict = {"protocol": "frozen strict and one-outer-fence semantic", "bootstrap_unit": "episode",
                    "bootstrap_samples": bootstrap_samples, "bootstrap_seed": 42,
                    "runs": {}, "contrasts": {}, "source_sha256": {},
                    "analysis_code_sha256": {name: text_sha256(root / name) for name in ANALYSIS_CODE}}
    result["source_sha256"].update({name: text_sha256(root / name) for name in FROZEN_FINAL_SOURCES})
    raw: dict[str, list[dict]] = {}
    for name, (run_id, schema) in RUNS.items():
        directory = root / "artifacts/v2/runs" / run_id
        verify_run(directory)
        path = directory / "predictions.jsonl"
        rows = _read_rows(path)
        assert_frozen_final_manifest(root, rows)
        raw[name] = rows
        strict = _evaluate(rows, schema, bootstrap_samples=bootstrap_samples)
        normalized = [{**row, "output": normalize_semantic_output(str(row["output"]))} for row in rows]
        semantic = _evaluate(normalized, schema, bootstrap_samples=bootstrap_samples)
        counts = _prediction_counts(normalized, schema)
        if strict["n"] != 54 or strict["gold_state_counts"] != {"nominal": 18, "failure": 18, "recovery": 18}:
            raise ValueError(f"not the frozen balanced 54-window final test: {name}")
        saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for key in PRIMARY_FIELDS:
            if abs(float(saved[key]) - float(strict[key])) > 1e-12:
                raise ValueError(f"saved strict metric differs from predictions: {name} {key}")
            if abs(float(saved["semantic_diagnostic"][key]) - float(semantic[key])) > 1e-12:
                raise ValueError(f"saved semantic metric differs from predictions: {name} {key}")
        result["runs"][name] = {"run_id": run_id, "schema": schema, "strict": strict,
                                "semantic": semantic, "semantic_predicted_state_counts": counts,
                                "outer_fence_removed_count": sum(str(a["output"]).strip() != b["output"] for a, b in zip(rows, normalized, strict=True))}
        for filename in ("predictions.jsonl", "metrics.json", "run_receipt.json", "config.json"):
            source = directory / filename
            result["source_sha256"][source.relative_to(root).as_posix()] = text_sha256(source)
    assert_same_final_references(raw)
    reference = raw["State Base"]
    for name in ("State SFT 42", "State SFT 43", "State SFT 44"):
        result["contrasts"][name] = compare(reference, raw[name], task_schema="state-only",
                                             semantic=True, samples=bootstrap_samples, seed=42)
    result["contrasts"]["Full SFT"] = compare(raw["Full Base"], raw["Full SFT"], task_schema="full",
                                                semantic=True, samples=bootstrap_samples, seed=42)
    return result


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total}"


def _decimal(value: float) -> str:
    return f"{value:.3f}"


def _interval(contrast: dict, endpoint: str) -> str:
    value = contrast["second_minus_first"][endpoint]
    lo, hi = value["ci95"]
    return f"{value['point']:+.3f} [{lo:+.3f},{hi:+.3f}]"


def render_final_assets(evidence: dict) -> dict[str, str]:
    if set(evidence["runs"]) != set(RUNS):
        raise ValueError("all six frozen Tier-B runs are required")
    table_rows = []
    bars = []
    labels = ("B", "S42", "S43", "S44", "FB", "FS42")
    for index, (name, label) in enumerate(zip(RUNS, labels, strict=True)):
        item = evidence["runs"][name]
        strict = item["strict"]
        semantic = item["semantic"]
        counts = item["semantic_predicted_state_counts"]
        nominal_fa = round(semantic["pre_failure_false_positive_rate"] * semantic["gold_state_counts"]["nominal"])
        table_rows.append(
            f"{name} & {_ratio(round(strict['json_valid_rate'] * strict['n']), strict['n'])}"
            f" & {_ratio(semantic['failure_correct'], semantic['failure_support'])}"
            f" & {_ratio(nominal_fa, semantic['gold_state_counts']['nominal'])}"
            f" & {_decimal(semantic['failure_precision'])} & {_decimal(semantic['state_macro_f1'])}"
            f" & {counts['nominal']}/{counts['failure']}/{counts['recovery']}/{counts['invalid']} " + r"\\"
        )
        x = 0.75 + index * 0.85
        for shift, value, style in ((-0.22, strict["json_valid_rate"], "draw=black,fill=white"),
                                    (0.0, semantic["failure_recall"], "draw=black,fill=black!65"),
                                    (0.22, semantic["pre_failure_false_positive_rate"],
                                     "draw=black,pattern=north east lines,pattern color=black")):
            bars.append(f"\\filldraw[{style}] ({x+shift:.2f},0) rectangle ({x+shift+0.17:.2f},{value:.4f});")
        bars.append(f"\\node[below,font=\\tiny] at ({x+0.08:.2f},0) {{{label}}};")
    table = "\n".join([
        "% Generated by scripts/generate_final_paper_assets.py from receipt-verified predictions.",
        r"\begin{tabular}{lrrrrrl}", r"\toprule",
        "Model & Strict JSON & F-rec & N-FA & F-prec & State-F1 & Pred N/F/R/I " + r"\\",
        r"\midrule", *table_rows, r"\bottomrule", r"\end{tabular}",
    ]) + "\n"
    paired_rows = []
    for name in ("State SFT 42", "State SFT 43", "State SFT 44"):
        contrast = evidence["contrasts"][name]
        paired_rows.append(f"{name} & {_interval(contrast, 'failure_recall')} & {_interval(contrast, 'state_macro_f1')} & {_interval(contrast, 'nominal_false_alarm_rate')} " + r"\\")
    paired = "\n".join([
        "% Paired six-episode bootstrap, candidate minus state-only Base; nominal FAs are not onset-relative.",
        r"\begin{tabular}{llll}", r"\toprule",
        "Candidate & $\\Delta$ F-rec [95\\% CI] & $\\Delta$ State-F1 [95\\% CI] & $\\Delta$ N-FA [95\\% CI] " + r"\\",
        r"\midrule", *paired_rows, r"\bottomrule", r"\end{tabular}",
    ]) + "\n"
    figure = "\n".join([
        "% Three distinct axes: protocol validity, semantic failure recall, nominal-window false alarms.",
        r"\begin{tikzpicture}[x=1cm,y=2.1cm]",
        r"\draw[->] (0.3,0) -- (5.95,0);",
        r"\draw[->] (0.4,0) -- (0.4,1.12);",
        r"\foreach \y in {0,0.5,1} {\draw (0.35,\y) -- (0.45,\y); \node[left,font=\tiny] at (0.35,\y) {\y};}",
        *bars,
        r"\node[anchor=west,font=\scriptsize] at (0.48,1.13) {"
        r"\tikz[baseline=-.5ex]\filldraw[draw=black,fill=white] (0,0) rectangle (.12,.1); JSON \quad "
        r"\tikz[baseline=-.5ex]\filldraw[draw=black,fill=black!65] (0,0) rectangle (.12,.1); F-rec \quad "
        r"\tikz[baseline=-.5ex]\filldraw[draw=black,pattern=north east lines,pattern color=black] (0,0) rectangle (.12,.1); N-FA};",
        r"\end{tikzpicture}",
    ]) + "\n"
    return {"tables/final.tex": table, "tables/paired_contrasts.tex": paired,
            "figures/final_tradeoff.tex": figure}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = load_final_evidence(root, bootstrap_samples=args.bootstrap_samples)
    assets = render_final_assets(evidence)
    for name, body in assets.items():
        path = root / "paper" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    evidence["generated_sha256"] = {
        name: text_sha256(root / "paper" / name)
        for name in assets
    }
    path = root / "artifacts/v2/final_paper_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "GENERATED", "runs": list(RUNS), "assets": sorted(assets), "bootstrap_unit": "episode"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
