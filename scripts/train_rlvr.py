from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.config import validate_rl_config
from reboot_recovery.evidence import finalize_run, write_json, write_run_start
from reboot_recovery.rewards import score_prediction, score_state_gated_prediction


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completion_text(completion: object) -> str:
    if isinstance(completion, list) and completion:
        completion = completion[-1]
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _class_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row["reference"]["failure_mode"]) for row in rows if row["reference"]["execution_state"] != "nominal")
    if not counts:
        return {}
    raw = {label: 1.0 / math.sqrt(count) for label, count in counts.items()}
    normalizer = sum(raw[label] * counts[label] for label in raw) / sum(counts.values())
    return {label: weight / normalizer for label, weight in raw.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gated GRPO/GSPO RLVR for structured REBOOT diagnosis")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-decision", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, choices=[2, 4], default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = validate_rl_config(_read_json(args.config))
    gate = _read_json(args.gate_decision)
    if gate.get("decision") != "RUN_RLVR":
        raise SystemExit(f"RLVR gate is closed: {gate.get('decision')}")
    if config["reward_scheme"] == "state_gated_v2" and (
        gate.get("protocol") != "V2" or gate.get("task_schema") != "full"
        or gate.get("verifier_validated") is not True
    ):
        raise SystemExit("V2 gate must verify a full-schema checkpoint and reward")
    data_path = args.dataset_root / f"{args.split}.jsonl"
    if not data_path.is_file():
        raise SystemExit(f"prepared split not found: {data_path}")
    if not args.sft_checkpoint.is_dir():
        raise SystemExit(f"SFT checkpoint not found: {args.sft_checkpoint}")
    if config["reward_scheme"] == "state_gated_v2":
        checkpoint_config = args.sft_checkpoint / "config.json"
        if not checkpoint_config.is_file() or _read_json(checkpoint_config).get("task_schema") != "full":
            raise SystemExit("V2 GRPO requires a full-schema SFT checkpoint")
    rows = _read_jsonl(data_path)
    missing = [str(args.dataset_root / image) for row in rows for image in row.get("images", []) if not (args.dataset_root / image).is_file()]
    if missing:
        raise SystemExit(f"missing prepared images; first={missing[:3]}")
    class_weights = (_class_weights(rows) if config["reward_scheme"] == "additive_v1"
                     and config.get("class_weighting", "inverse_sqrt") == "inverse_sqrt" else {})
    phase_labels = {str(row["reference"]["phase_name"]).casefold() for row in rows}
    failure_mode_labels = {
        str(row["reference"]["failure_mode"]).casefold()
        for row in rows
        if str(row["reference"]["failure_mode"]).casefold() != "none"
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "artifacts" / "runs" / f"{config['algorithm']}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved = dict(config)
    if args.max_steps is not None:
        resolved["max_steps"] = args.max_steps
    if args.num_generations is not None:
        resolved["num_generations"] = args.num_generations
    resolved.update({
        "stage": "rlvr", "base_model": str(args.model_root or args.base_model),
        "sft_checkpoint": str(args.sft_checkpoint.resolve()), "dataset_root": str(args.dataset_root.resolve()),
        "dataset_revision": args.dataset_revision, "model_revision": args.model_revision,
        "failure_mode_class_weights": class_weights, "seed": args.seed,
    })
    receipt = write_run_start(
        output_dir, root=ROOT, run_id=output_dir.name, config=resolved,
        dataset_revision=args.dataset_revision, model_revision=args.model_revision, seed=args.seed,
    )
    if args.dry_run:
        summary = {
            "status": "DRY_RUN_VERIFIED", "algorithm": config["algorithm"],
            "reward_scheme": config["reward_scheme"],
            "importance_sampling_level": config["importance_sampling_level"], "loss_type": config["loss_type"],
            "rows": len(rows), "class_weights": class_weights,
            "max_steps": resolved["max_steps"], "num_generations": resolved["num_generations"],
        }
        write_json(output_dir / "dry_run.json", summary)
        finalize_run(output_dir, receipt, status="DRY_RUN_VERIFIED", outputs=[output_dir / "dry_run.json"])
        print(json.dumps(summary, indent=2))
        return 0

    import torch
    if not torch.cuda.is_available():
        finalize_run(output_dir, receipt, status="BLOCKED_NO_CUDA")
        raise SystemExit("CUDA GPU is required for RLVR.")
    from datasets import Dataset
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    model_source = str(args.model_root or args.base_model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_source, revision=args.model_revision, local_files_only=args.model_root is not None,
        torch_dtype=dtype, device_map="auto", quantization_config=quantization,
    )
    model = PeftModel.from_pretrained(base, str(args.sft_checkpoint), is_trainable=True)
    processor = AutoProcessor.from_pretrained(
        model_source, revision=args.model_revision, local_files_only=args.model_root is not None,
    )
    materialized = []
    for row in rows:
        copy = {"prompt": row["prompt"], "reference": row["reference"]}
        images = []
        for image_path in row["images"]:
            with Image.open(args.dataset_root / image_path) as image:
                images.append(image.convert("RGB").copy())
        copy["images"] = images
        materialized.append(copy)

    def phase_reward(completions: list[object], reference: list[dict[str, Any]], **_: object) -> list[float]:
        return [
            score_prediction(
                ref, completion_text(output), phase_labels=phase_labels,
                failure_mode_labels=failure_mode_labels,
            ).phase
            for output, ref in zip(completions, reference, strict=True)
        ]

    def state_reward(completions: list[object], reference: list[dict[str, Any]], **_: object) -> list[float]:
        return [
            score_prediction(
                ref, completion_text(output), phase_labels=phase_labels,
                failure_mode_labels=failure_mode_labels,
            ).state
            for output, ref in zip(completions, reference, strict=True)
        ]

    def failure_reward(completions: list[object], reference: list[dict[str, Any]], **_: object) -> list[float]:
        rewards = []
        for output, ref in zip(completions, reference, strict=True):
            score = score_prediction(
                ref, completion_text(output), phase_labels=phase_labels,
                failure_mode_labels=failure_mode_labels,
            )
            rewards.append(score.failure_mode * class_weights.get(str(ref["failure_mode"]), 1.0))
        return rewards

    def state_gated_reward(completions: list[object], reference: list[dict[str, Any]], **_: object) -> list[float]:
        return [
            score_state_gated_prediction(
                ref, completion_text(output), phase_labels=phase_labels,
                failure_mode_labels=failure_mode_labels,
                reward_config=config["state_gated_reward"],
            )
            for output, ref in zip(completions, reference, strict=True)
        ]

    training = GRPOConfig(
        output_dir=str(output_dir), max_steps=int(resolved["max_steps"]), learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_generations=int(resolved["num_generations"]), max_completion_length=int(config["max_completion_length"]),
        importance_sampling_level=str(config["importance_sampling_level"]), loss_type=str(config["loss_type"]),
        reward_weights=[float(value) for value in config["reward_weights"]], beta=float(config["beta"]),
        logging_steps=1, save_steps=25, save_strategy="steps", report_to="none",
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model, args=training, processing_class=processor,
        reward_funcs=([state_gated_reward] if config["reward_scheme"] == "state_gated_v2" else [phase_reward, state_reward, failure_reward]),
        train_dataset=Dataset.from_list(materialized),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    started = time.perf_counter()
    trainer.train()
    wall_clock = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    log_path = output_dir / "train_log.json"
    write_json(log_path, {"wall_clock_seconds": wall_clock, "trainable_parameters": trainable, "log_history": trainer.state.log_history})
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, {
        "wall_clock_seconds": wall_clock,
        "trainable_parameters": trainable,
        "final_log": trainer.state.log_history[-1] if trainer.state.log_history else {},
        "global_step": trainer.state.global_step,
    })
    finalize_run(
        output_dir, receipt, status="COMPLETED", outputs=[log_path, metrics_path],
        vram_peak_bytes=int(torch.cuda.max_memory_allocated()), trainable_parameters=trainable,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
