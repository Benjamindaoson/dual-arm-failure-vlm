"""Recheck frozen submission artifacts without changing experimental semantics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_episode_exposure_ledger import validate_ledger
from scripts.build_paper import verify_build_receipt
from scripts.generate_final_paper_assets import (
    RUNS, _evaluate, _read_rows, load_final_evidence, render_final_assets, text_sha256,
)
from scripts.verify_v2_evidence import verify_run
from reboot_recovery.semantic_parser import normalize_semantic_output


def verify_hash_map(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or text_sha256(path) != expected:
            raise ValueError(f"missing or changed artifact: {path}")


def verify_recomputed_evidence(saved: dict, regenerated: dict) -> None:
    """Compare every figure/table input, not just a headline count."""
    if {key: value for key, value in saved.items() if key != "generated_sha256"} != regenerated:
        raise ValueError("saved final evidence differs from full recomputation")


def assert_matched_schema_inputs(state_config: dict, full_config: dict,
                                 state_rows: list[dict], full_rows: list[dict]) -> None:
    """Reject an apparent schema contrast with any changed checkpoint or observation."""
    differences = {key for key in set(state_config) | set(full_config)
                   if state_config.get(key) != full_config.get(key)}
    if differences != {"task_schema"} or state_config.get("task_schema") != "state-only" \
            or full_config.get("task_schema") != "full" or state_config.get("adapter") is not None \
            or state_config.get("stage") != "base_eval":
        raise ValueError("schema contrast configuration differs beyond task_schema")
    if (not state_rows or len(state_rows) != len(full_rows)
            or len({row["id"] for row in state_rows}) != len(state_rows)
            or len({row["id"] for row in full_rows}) != len(full_rows)):
        raise ValueError("schema contrast IDs are empty, duplicated, or differ")
    for left, right in zip(state_rows, full_rows, strict=True):
        if (left["id"] != right["id"] or left["reference"] != right["reference"]
                or left.get("split_receipt_sha256") != right.get("split_receipt_sha256")):
            raise ValueError("schema contrast reference or sample ID differs")


def verify_schema_contrast(root: Path) -> dict:
    """Recompute the fixed-Base C2 output-schema contrast from completed runs."""
    names = ("c2-base-val", "multitask-base-val")
    directories = [root / "artifacts/v2/runs" / name for name in names]
    for directory in directories:
        verify_run(directory)
    configs = [json.loads((directory / "config.json").read_text(encoding="utf-8"))
               for directory in directories]
    rows = [_read_rows(directory / "predictions.jsonl") for directory in directories]
    assert_matched_schema_inputs(*configs, *rows)
    split_receipt = root / "artifacts/splits/split_receipt.json"
    split_hash = text_sha256(split_receipt)
    if any(row.get("split_receipt_sha256") != split_hash or row["reference"].get("split") != "val"
           for run_rows in rows for row in run_rows):
        raise ValueError("schema contrast does not match frozen validation split receipt")
    recomputed = []
    for directory, run_rows, schema in zip(directories, rows, ("state-only", "full"), strict=True):
        normalized = [{**row, "output": normalize_semantic_output(str(row["output"]))}
                      for row in run_rows]
        measured = _evaluate(normalized, schema, bootstrap_samples=1000)
        saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["semantic_diagnostic"]
        for key in ("n", "failure_correct", "failure_support", "failure_recall",
                    "state_macro_f1", "failure_precision", "pre_failure_false_positive_rate"):
            if abs(float(measured[key]) - float(saved[key])) > 1e-12:
                raise ValueError(f"schema contrast saved semantic metric differs: {directory.name} {key}")
        recomputed.append(measured)
    sources = {"artifacts/splits/split_receipt.json": split_hash}
    for directory in directories:
        for filename in ("config.json", "metrics.json", "predictions.jsonl", "run_receipt.json"):
            path = directory / filename
            sources[path.relative_to(root).as_posix()] = text_sha256(path)
    return {
        "run_ids": list(names), "sample_count": len(rows[0]),
        "failure_support": recomputed[0]["failure_support"],
        "state_only_semantic_failure_correct": recomputed[0]["failure_correct"],
        "full_schema_semantic_failure_correct": recomputed[1]["failure_correct"],
        "config_differences": ["task_schema"], "source_sha256": sources,
    }


def webp_pdf_blockers(manuscript: str, *, pdf_author: str | None) -> list[str]:
    """Check the official default CoRL mode for WEBP double-blind review."""
    blockers = []
    style = re.search(r"\\usepackage(?:\[([^\]]+)\])?\{corl_2026\}", manuscript)
    if style is None or style.group(1) is not None:
        blockers.append("double-blind style")
    if not re.search(r"\\author\s*\{\s*\}", manuscript):
        blockers.append("source author identity")
    if pdf_author != "Anonymous Submission":
        blockers.append("PDF author metadata")
    return blockers


def openreview_form_blockers(confirmation: dict | None) -> list[str]:
    """Keep private author-form facts separate from the anonymous paper source."""
    authors = confirmation.get("authors") if isinstance(confirmation, dict) else None
    valid = (isinstance(authors, list) and bool(authors)
             and confirmation.get("confirmed_by_author") is True
             and confirmation.get("email_sharing_confirmed") is True
             and confirmation.get("data_release_confirmed") is True
             and all(isinstance(item, dict) and item.get("order") == index
                     and isinstance(item.get("name"), str) and item["name"].strip()
                     and isinstance(item.get("profile_id"), str)
                     and re.fullmatch(r"~\S+", item["profile_id"])
                     for index, item in enumerate(authors, 1)))
    return [] if valid else ["OpenReview form"]


def verify_rl_gate(root: Path) -> dict:
    path = root / "artifacts/v2/final_rl_gate_decision.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "VALIDATION_GATE", "protocol": "V2",
        "decision": "REVISIT_REPRESENTATION_OR_SUPERVISION",
        "base_failure_correct": 0, "sft_failure_correct": 0,
        "failure_correct_gain": 0, "verifier_validated": True,
    }
    if any(gate.get(key) != expected for key, expected in required.items()):
        raise ValueError("RL gate is not the frozen closed validation gate")
    runs_root = root / "artifacts/v2/runs"
    for key, run_id in (("base_predictions_sha256", "final-base-full-val"),
                        ("sft_predictions_sha256", "final-sft-full-seed42-val")):
        predictions = runs_root / run_id / "predictions.jsonl"
        raw_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
        if gate.get(key) not in (raw_hash, text_sha256(predictions)):
            raise ValueError(f"RL gate validation source differs: {run_id}")
    if runs_root.is_dir():
        for config_path in runs_root.glob("*/config.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if str(config.get("stage", "")).lower() in {"rlvr", "grpo", "gspo"}:
                raise ValueError(f"V2 RL run exists despite closed gate: {config_path.parent.name}")
    return gate


def _adapter_hashes(root: Path) -> dict[str, str]:
    verified = {}
    for name in ("final-sft-state-seed42", "final-sft-state-seed43",
                 "final-sft-state-seed44", "final-sft-full-seed42"):
        receipt_path = root / "artifacts/v2/runs" / name / "run_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = receipt["output_sha256"]["adapter_model.safetensors"]
        path = root / "checkpoints/v2-final/artifacts/v2/runs" / name / "adapter_model.safetensors"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"final adapter differs from training receipt: {name}")
        verified[name] = expected
    return verified


def _pdf_pages(path: Path) -> int | None:
    executable = shutil.which("pdfinfo")
    if not executable or not path.is_file():
        return None
    result = subprocess.run([executable, str(path)], capture_output=True, text=True, errors="replace", check=True)
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("pdfinfo did not report page count")


def _pdf_author(path: Path) -> str | None:
    executable = shutil.which("pdfinfo")
    if not executable or not path.is_file():
        return None
    result = subprocess.run([executable, str(path)], capture_output=True, text=True, errors="replace", check=True)
    for line in result.stdout.splitlines():
        if line.startswith("Author:"):
            return line.split(":", 1)[1].strip()
    return ""


def _candidate_metadata(root: Path) -> dict[str, dict]:
    candidates = {}
    for name in ("usbc_recovery_install", "rj45_recovery_install", "m12_recovery_install"):
        directory = root / "artifacts/v2/official_metadata" / name
        parquet = directory / "episodes.parquet"
        raw = parquet.read_bytes() if parquet.is_file() else b""
        candidates[name] = {
            "info_json_present": (directory / "info.json").is_file(),
            "phase_annotations_present": (directory / "meta/phase.json").is_file(),
            "episodes_parquet_complete_magic": raw.startswith(b"PAR1") and raw.endswith(b"PAR1"),
            "videos_present": any(directory.rglob("*.mp4")) if directory.is_dir() else False,
        }
    return candidates


def audit_submission(root: Path = ROOT) -> dict:
    root = root.resolve()
    exposure = validate_ledger(root, root / "artifacts/episode_exposure_ledger.csv")
    saved_exposure = json.loads((root / "artifacts/episode_exposure_ledger_receipt.json").read_text(encoding="utf-8"))
    if saved_exposure != exposure:
        raise ValueError("episode exposure receipt differs from rebuilt ledger and source hashes")
    runs_root = root / "artifacts/v2/runs"
    receipts = [
        verify_run(path) for path in sorted(runs_root.iterdir())
        if path.is_dir() and (path / "run_receipt.json").is_file()
        and json.loads((path / "run_receipt.json").read_text(encoding="utf-8")).get("status") == "COMPLETED"
    ]
    if len(receipts) < 44:
        raise ValueError("fewer than 44 completed V2 runs verify")
    saved = json.loads((root / "artifacts/v2/final_paper_evidence.json").read_text(encoding="utf-8"))
    verify_hash_map(root, saved["source_sha256"])
    verify_hash_map(root, saved["analysis_code_sha256"])
    verify_hash_map(root / "paper", saved["generated_sha256"])
    regenerated = load_final_evidence(root, bootstrap_samples=int(saved["bootstrap_samples"]))
    generated = render_final_assets(regenerated)
    for relative, body in generated.items():
        if (root / "paper" / relative).read_text(encoding="utf-8") != body:
            raise ValueError(f"paper asset does not reproduce from predictions: {relative}")
    verify_recomputed_evidence(saved, regenerated)
    adapters = _adapter_hashes(root)
    manuscript = (root / "paper/main.tex").read_text(encoding="utf-8")
    if not all(term in manuscript.lower() for term in ("internal evidence", "no v2 grpo or gspo was run",
                                                       "nominal false alarms", r"\input{tables/final.tex}")):
        raise ValueError("manuscript omits an evidence limitation or generated table")
    if not re.search(r"\\usepackage(?:\[[^\]]+\])?\{corl_2026\}", manuscript):
        raise ValueError("official CoRL 2026 submission style not loaded")
    pdf = root / "outputs/v2/paper_build/main.pdf"
    pages = _pdf_pages(pdf)
    paper_build = verify_build_receipt(root, pages)
    if not 1 <= paper_build["main_pages"] <= 4:
        raise ValueError(f"WEBP main text exceeds four pages: {paper_build['main_pages']}")
    schema_contrast = verify_schema_contrast(root)
    gate = verify_rl_gate(root)
    candidates = _candidate_metadata(root)
    second_task_local_ready = any(all(value.values()) for value in candidates.values())
    confirmation_path = root / "outputs/v2/openreview_confirmation.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8")) if confirmation_path.is_file() else None
    pdf_blockers = webp_pdf_blockers(manuscript, pdf_author=_pdf_author(pdf))
    if pages is None:
        pdf_blockers.append("PDF page count not independently verified")
    form_blockers = openreview_form_blockers(confirmation)
    return {
        "evidence_status": "PASS_WITH_LIMITATIONS",
        "pdf_ready": not pdf_blockers,
        "pdf_blockers": pdf_blockers,
        "submission_ready": not pdf_blockers and not form_blockers,
        "submission_blockers": pdf_blockers + form_blockers,
        "official_venue": {
            "url": "https://beneath-the-policy.github.io/",
            "format": "at most 4 main-text pages; references/appendix excluded; CoRL 2026 submission template",
            "review": "double-blind",
            "deadline": "2026-10-09 23:59 Central Time",
        },
        "completed_v2_runs_verified": len(receipts),
        "final_run_ids": {name: RUNS[name][0] for name in RUNS},
        "final_prediction_source_sha256": saved["source_sha256"],
        "analysis_code_sha256": saved["analysis_code_sha256"],
        "generated_asset_sha256": saved["generated_sha256"],
        "adapter_hashes_verified": len(adapters),
        "adapter_sha256": adapters,
        "fixed_checkpoint_schema_contrast": schema_contrast,
        "episode_exposure": {key: exposure[key] for key in ("episodes", "usable", "untouched_same_task",
                                                            "internal_tier_b", "ledger_sha256")},
        "independent_evidence": {
            "route1_same_task": "NO_UNTOUCHED_EPISODES_IN_LOCAL_60_EPISODE_SAMPLE",
            "route2_second_task": "NOT_EXECUTED_NO_LOCAL_LABEL_GROUNDED_VIDEO_OR_GPU"
                if not second_task_local_ready else "CANDIDATE_METADATA_READY_ONLY",
            "candidate_local_metadata": candidates,
            "claim": "internal single-task diagnostic; no independent or cross-task replication",
        },
        "local_compute": {
            "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
            "project_venv_torch_installed": importlib.util.find_spec("torch") is not None,
            "base_qwen_model_local": (root / "models/Qwen2.5-VL-3B-Instruct").is_dir(),
        },
        "rl_gate": gate["decision"] + "; no V2 GRPO or GSPO",
        "rl_gate_source_sha256": {
            "artifacts/v2/final_rl_gate_decision.json": text_sha256(root / "artifacts/v2/final_rl_gate_decision.json"),
            "artifacts/v2/runs/final-base-full-val/predictions.jsonl": text_sha256(root / "artifacts/v2/runs/final-base-full-val/predictions.jsonl"),
            "artifacts/v2/runs/final-sft-full-seed42-val/predictions.jsonl": text_sha256(root / "artifacts/v2/runs/final-sft-full-seed42-val/predictions.jsonl"),
        },
        "claim_hierarchy": {
            "evaluation_axis_sensitivity": "SUPPORTED_INTERNAL",
            "fixed_base_schema_contrast": "SUPPORTED_VALIDATION_DIAGNOSTIC",
            "post_training_operating_points": "SUPPORTED_RECIPE_SPECIFIC_ONLY",
            "reporting_checklist": "PROPOSED_NOT_UNIVERSAL",
        },
        "pdf_pages": pages,
        "main_text_pages": paper_build["main_pages"],
        "pdf_sha256": paper_build["pdf_sha256"],
        "paper_build_git_commit": paper_build["git_commit"],
        "paper_build_source_date_epoch": paper_build["source_date_epoch"],
        "paper_build_source_sha256": paper_build["source_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit_submission(root)
    (root / "artifacts/final_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("evidence_status", "pdf_ready",
                                                  "submission_ready", "submission_blockers",
                                                  "pdf_pages", "main_text_pages")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
