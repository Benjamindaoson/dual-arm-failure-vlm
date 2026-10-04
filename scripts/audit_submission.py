"""Recheck frozen submission artifacts without changing experimental semantics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_episode_exposure_ledger import validate_ledger
from scripts.build_paper import verify_build_receipt
from scripts.generate_final_paper_assets import RUNS, load_final_evidence, render_final_assets, text_sha256
from scripts.verify_v2_evidence import verify_run


def verify_hash_map(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or text_sha256(path) != expected:
            raise ValueError(f"missing or changed artifact: {path}")


def verify_recomputed_evidence(saved: dict, regenerated: dict) -> None:
    """Compare every figure/table input, not just a headline count."""
    if {key: value for key, value in saved.items() if key != "generated_sha256"} != regenerated:
        raise ValueError("saved final evidence differs from full recomputation")


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
    if not all(term in manuscript for term in ("internal evidence", "No V2 GRPO or GSPO was run",
                                               "nominal false alarms", r"\input{tables/final.tex}")):
        raise ValueError("manuscript omits an evidence limitation or generated table")
    if r"\usepackage{corl_2026}" not in manuscript:
        raise ValueError("official CoRL 2026 submission style not loaded")
    pdf = root / "outputs/v2/paper_build/main.pdf"
    pages = _pdf_pages(pdf)
    if pages is not None and not 2 <= pages <= 4:
        raise ValueError(f"workshop length out of range: {pages}")
    paper_build = verify_build_receipt(root, pages)
    gate = verify_rl_gate(root)
    candidates = _candidate_metadata(root)
    second_task_local_ready = any(all(value.values()) for value in candidates.values())
    blockers = []
    if "Author information pending" in manuscript or r"\usepackage[final]{corl_2026}" not in manuscript:
        blockers.append("author metadata")
    if pages is None:
        blockers.append("PDF page count not independently verified")
    return {
        "evidence_status": "PASS_WITH_LIMITATIONS",
        "submission_ready": not blockers,
        "submission_blockers": blockers,
        "official_venue": {
            "url": "https://umi-arena.airoa.io/",
            "format": "2-4 page extended abstract, CoRL main-conference template",
            "review": "single-blind",
            "deadline": "2026-10-16",
        },
        "completed_v2_runs_verified": len(receipts),
        "final_run_ids": {name: RUNS[name][0] for name in RUNS},
        "final_prediction_source_sha256": saved["source_sha256"],
        "analysis_code_sha256": saved["analysis_code_sha256"],
        "generated_asset_sha256": saved["generated_sha256"],
        "adapter_hashes_verified": len(adapters),
        "adapter_sha256": adapters,
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
            "H1": "SUPPORTED_INTERNAL",
            "H2": "SUPPORTED_RECIPE_SPECIFIC_ONLY",
            "H3": "UNTESTED_INTERVENTION",
        },
        "pdf_pages": pages,
        "pdf_sha256": paper_build["pdf_sha256"],
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
    print(json.dumps({key: report[key] for key in ("evidence_status", "submission_ready",
                                                  "submission_blockers", "pdf_pages")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
