"""Reconstruct episode exposure from frozen manifests; do not infer independence from a refit split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


SOURCE_PATHS = (
    "artifacts/data_audit/source/meta/info.json",
    "artifacts/splits/split_receipt.json",
    "artifacts/manifests/reboot_windows.jsonl",
    "artifacts/v2/manifests/dense_windows.jsonl",
    "artifacts/v2/manifests/final_sparse.jsonl",
    "artifacts/v2/splits/paper_final_test_receipt.json",
)
DIAGNOSTIC_RUNS = {
    "camera_screen_participation": ("c0-sft-v1-diagnostic", "c1-sft-v1-diagnostic", "c2-sft-v1-diagnostic"),
    "dense_supervision_participation": ("dense-sft-v1-diagnostic",),
    "trace_text_participation": ("trace-sft-v1-diagnostic",),
    "schema_ablation_participation": ("multitask-sft-v1-diagnostic",),
}
FIELDNAMES = (
    "episode_id", "task", "usable", "v1_split", "v2_development_participation",
    "camera_screen_participation", "dense_supervision_participation",
    "trace_text_participation", "schema_ablation_participation",
    "final_refit_train", "final_refit_validation", "final_tier_b_test",
    "documented_decision_influence", "other_decision_influence", "untouched_same_task",
    "evidence_tier",
)


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _json(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _rows(root: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def _roles(rows: list[dict]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in rows:
        episode = str(row["episode_index"])
        role = str(row["split"])
        if episode in roles and roles[episode] != role:
            raise ValueError(f"episode appears in multiple splits: {episode}")
        roles[episode] = role
    return roles


def _diagnostic_ids(root: Path, runs: tuple[str, ...]) -> set[str]:
    ids: set[str] = set()
    for name in runs:
        path = root / "artifacts/v2/runs" / name / "predictions.jsonl"
        if not path.is_file():
            raise ValueError(f"expected diagnostic predictions absent: {path}")
        ids.update(str(row["reference"]["episode_index"]) for row in _rows(root, path.relative_to(root).as_posix()))
    return ids


def build_ledger(root: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    root = root.resolve()
    info = _json(root, SOURCE_PATHS[0])
    v1 = _json(root, SOURCE_PATHS[1])
    v1_roles = _roles(_rows(root, SOURCE_PATHS[2]))
    dense_roles = _roles(_rows(root, SOURCE_PATHS[3]))
    final_roles = _roles(_rows(root, SOURCE_PATHS[4]))
    final = _json(root, SOURCE_PATHS[5])
    n = int(info["total_episodes"])
    original = {f"{index:02d}" for index in range(n)}
    quarantine = set(v1["quarantined_episode_ids"])
    usable = original - quarantine
    v1_receipt_roles = {episode: role for role, episodes in v1["episodes"].items() for episode in episodes}
    if set(v1_roles) != usable or v1_roles != v1_receipt_roles:
        raise ValueError("V1 window roles do not match complete-episode split receipt")
    if dense_roles != v1_roles:
        raise ValueError("dense supervision does not match frozen V1 development split")
    if set(final_roles) != set(final["episode_to_split"]) or final_roles != final["episode_to_split"]:
        raise ValueError("final sparse window roles do not match frozen final receipt")
    if set(final_roles) != set(v1["episodes"]["train"]):
        raise ValueError("final refit pool is not exactly prior V1 training pool")
    for role in ("train", "val", "test"):
        if {e for e, value in final_roles.items() if value == role} != set(final["episodes"][role]):
            raise ValueError(f"final {role} assignments disagree with receipt")
    diagnostic = {field: _diagnostic_ids(root, runs) for field, runs in DIAGNOSTIC_RUNS.items()}
    if any(not ids <= set(v1["episodes"]["test"]) for ids in diagnostic.values()):
        raise ValueError("a diagnostic run contains a non-V1-test episode")
    dev_ids = set(v1["episodes"]["train"]) | set(v1["episodes"]["val"])
    result: list[dict[str, str]] = []
    for episode in sorted(original):
        valid = episode in usable
        v1_role = v1_roles.get(episode, "quarantined")
        final_role = final_roles.get(episode, "none")
        if v1_role == "val":
            influence = "V1 validation; V2 camera/supervision/schema method-selection validation"
        elif v1_role == "test":
            influence = "V1 retrospective diagnostic inspection"
        elif v1_role == "train":
            influence = "V1/V2 model fitting; individual prompt/hyperparameter influence not logged"
        else:
            influence = "none documented; quarantined"
        row = {
            "episode_id": episode,
            "task": "16mm-cylinder-install",
            "usable": "yes" if valid else "no",
            "v1_split": v1_role,
            "v2_development_participation": v1_role if episode in dev_ids else "none",
            "camera_screen_participation": v1_role if episode in dev_ids else ("diagnostic_test" if episode in diagnostic["camera_screen_participation"] else "none"),
            "dense_supervision_participation": v1_role if episode in dev_ids else ("diagnostic_test" if episode in diagnostic["dense_supervision_participation"] else "none"),
            "trace_text_participation": v1_role if episode in dev_ids else ("diagnostic_test" if episode in diagnostic["trace_text_participation"] else "none"),
            "schema_ablation_participation": v1_role if episode in dev_ids else ("diagnostic_test" if episode in diagnostic["schema_ablation_participation"] else "none"),
            "final_refit_train": "yes" if final_role == "train" else "no",
            "final_refit_validation": "yes" if final_role == "val" else "no",
            "final_tier_b_test": "yes" if final_role == "test" else "no",
            "documented_decision_influence": influence,
            "other_decision_influence": "unknown; no exhaustive human decision log",
            "untouched_same_task": "no" if valid else "not_eligible",
            "evidence_tier": "internal_exposed_test" if final_role == "test" else ("quarantined" if not valid else "development_exposed"),
        }
        result.append(row)
    source_names = list(SOURCE_PATHS)
    source_names.extend(f"artifacts/v2/runs/{run}/predictions.jsonl" for runs in DIAGNOSTIC_RUNS.values() for run in runs)
    sources = {name: _text_sha256(root / name) for name in source_names}
    return result, sources


def _csv_body(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_csv_body(rows), encoding="utf-8", newline="")


def validate_ledger(root: Path, path: Path) -> dict[str, object]:
    expected, sources = build_ledger(root)
    if path.read_text(encoding="utf-8") != _csv_body(expected):
        raise ValueError("exposure ledger differs from frozen source manifests")
    return {
        "status": "VERIFIED",
        "episodes": len(expected),
        "usable": sum(row["usable"] == "yes" for row in expected),
        "untouched_same_task": sum(row["untouched_same_task"] == "yes" for row in expected),
        "internal_tier_b": [row["episode_id"] for row in expected if row["final_tier_b_test"] == "yes"],
        "ledger_sha256": _text_sha256(path),
        "source_sha256": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    ledger = args.ledger or root / "artifacts/episode_exposure_ledger.csv"
    if not args.validate_only:
        rows, _ = build_ledger(root)
        write_ledger(ledger, rows)
    receipt = validate_ledger(root, ledger)
    receipt_path = root / "artifacts/episode_exposure_ledger_receipt.json"
    if not args.validate_only:
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "episodes", "usable", "untouched_same_task", "internal_tier_b")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
