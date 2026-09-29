from __future__ import annotations

from datetime import datetime, timezone
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets", "lerobot"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_runtime: dict[str, Any] = {"available": False, "cuda_available": False, "cuda_version": None}
    try:
        import torch
        torch_runtime = {
            "available": True,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": torch_runtime,
        "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ,
        "gpu": _command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
    }


def git_commit(root: str | Path) -> str | None:
    return _command("git", "-C", str(Path(root)), "rev-parse", "HEAD")


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finalize_abandoned_run(receipt_path: Path) -> None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "RUNNING":
            receipt.update({"status": "FAILED_OR_INTERRUPTED", "end_time": utc_now()})
            write_json(receipt_path, receipt)
    except (OSError, json.JSONDecodeError):
        pass


def write_run_start(
    run_dir: str | Path,
    *,
    root: str | Path,
    run_id: str,
    config: dict[str, Any],
    dataset_revision: str | None,
    model_revision: str | None,
    seed: int,
) -> dict[str, Any]:
    directory = Path(run_dir)
    environment = environment_snapshot()
    receipt = {
        "run_id": run_id,
        "status": "RUNNING",
        "git_commit": git_commit(root),
        "config_sha256": json_sha256(config),
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "seed": seed,
        "start_time": utc_now(),
        "end_time": None,
        "vram_peak_bytes": None,
        "trainable_parameters": None,
        "output_sha256": {},
        "gpu": environment["torch_runtime"].get("gpu_name"),
        "vram_total_bytes": environment["torch_runtime"].get("vram_bytes"),
        "cuda": environment["torch_runtime"].get("cuda_version"),
        "torch": environment["packages"].get("torch"),
        "transformers": environment["packages"].get("transformers"),
        "trl": environment["packages"].get("trl"),
        "peft": environment["packages"].get("peft"),
        "bitsandbytes": environment["packages"].get("bitsandbytes"),
    }
    write_json(directory / "config.json", config)
    write_json(directory / "environment.json", environment)
    (directory / "stdout.log").touch()
    (directory / "stderr.log").touch()
    write_json(directory / "run_receipt.json", receipt)
    atexit.register(_finalize_abandoned_run, directory / "run_receipt.json")
    return receipt


def finalize_run(
    run_dir: str | Path,
    receipt: dict[str, Any],
    *,
    status: str,
    outputs: Iterable[str | Path] = (),
    vram_peak_bytes: int | None = None,
    trainable_parameters: int | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    end_time = utc_now()
    start = datetime.fromisoformat(str(receipt["start_time"]))
    end = datetime.fromisoformat(end_time)
    wall_clock_seconds = (end - start).total_seconds()
    timing_path = directory / "timing.json"
    memory_path = directory / "memory.json"
    write_json(timing_path, {
        "start_time": receipt["start_time"], "end_time": end_time,
        "wall_clock_seconds": wall_clock_seconds,
    })
    write_json(memory_path, {
        "vram_total_bytes": receipt.get("vram_total_bytes"),
        "vram_peak_bytes": vram_peak_bytes,
    })
    all_outputs = [*outputs, timing_path, memory_path]
    hashes = {str(Path(path).relative_to(directory)): file_sha256(path) for path in all_outputs if Path(path).is_file()}
    receipt.update({
        "status": status,
        "end_time": end_time,
        "wall_clock_seconds": wall_clock_seconds,
        "vram_peak_bytes": vram_peak_bytes,
        "trainable_parameters": trainable_parameters,
        "output_sha256": hashes,
    })
    write_json(directory / "run_receipt.json", receipt)
    return receipt
