from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.checkpoints import resolve_checkpoint
from reboot_recovery.evidence import finalize_run, write_json, write_run_start
from reboot_recovery.prompts import to_state_only_record


LANGUAGE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT for the REBOOT failure-aware execution critic")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--variant", choices=["SFT-Language", "SFT-VisionLanguage"], default="SFT-Language")
    parser.add_argument("--task-schema", choices=["full", "state-only"], default="full")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value for bounded smoke runs; -1 uses epochs")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    missing = [str(root / relative) for row in rows for relative in row.get("images", []) if not (root / relative).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} prepared images; first={missing[:3]}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_path = args.dataset_root / f"{args.train_split}.jsonl"
    eval_path = args.dataset_root / f"{args.eval_split}.jsonl"
    if not train_path.is_file() or not eval_path.is_file():
        raise SystemExit(f"prepared splits not found: {train_path}, {eval_path}")
    train_rows, eval_rows = _read_jsonl(train_path), _read_jsonl(eval_path)
    _validate_rows(args.dataset_root, train_rows + eval_rows)
    if args.task_schema == "state-only":
        train_rows = [to_state_only_record(row) for row in train_rows]
        eval_rows = [to_state_only_record(row) for row in eval_rows]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "artifacts" / "runs" / f"sft-{args.variant.lower()}-{stamp}"
    checkpoint = resolve_checkpoint(output_dir, args.resume_from_checkpoint)
    if output_dir.exists() and checkpoint is None:
        raise SystemExit(f"output directory already exists and no checkpoint was selected: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = str(args.model_root or args.model)
    target_modules: str | list[str] = LANGUAGE_TARGETS if args.variant == "SFT-Language" else "all-linear"
    config = {
        "stage": "sft", "variant": args.variant, "model": model_source,
        "task_schema": args.task_schema, "dataset_root": str(args.dataset_root.resolve()),
        "model_revision": args.model_revision, "dataset_revision": args.dataset_revision,
        "train_split": args.train_split, "eval_split": args.eval_split,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.grad_accum, "quantization": "4-bit NF4",
        "lora": {"rank": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target_modules": target_modules},
        "gradient_checkpointing": True, "completion_only_loss": True, "seed": args.seed,
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
    }
    receipt = write_run_start(
        output_dir, root=ROOT, run_id=output_dir.name, config=config,
        dataset_revision=args.dataset_revision, model_revision=args.model_revision, seed=args.seed,
    )
    if args.dry_run:
        summary = {"status": "DRY_RUN_VERIFIED", "train_rows": len(train_rows), "eval_rows": len(eval_rows), "variant": args.variant}
        write_json(output_dir / "dry_run.json", summary)
        finalize_run(output_dir, receipt, status="DRY_RUN_VERIFIED", outputs=[output_dir / "dry_run.json"])
        print(json.dumps(summary, indent=2))
        return 0

    import torch
    if not torch.cuda.is_available():
        finalize_run(output_dir, receipt, status="BLOCKED_NO_CUDA")
        raise SystemExit("CUDA GPU is required for the SFT pilot.")
    from datasets import Dataset
    from peft import LoraConfig
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    def materialize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            copy = dict(row)
            images = []
            for path in row["images"]:
                with Image.open(args.dataset_root / path) as image:
                    images.append(image.convert("RGB").copy())
            copy["images"] = images
            result.append(copy)
        return result

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_source, revision=args.model_revision, local_files_only=args.model_root is not None,
        torch_dtype=dtype, device_map="auto", quantization_config=quantization,
    )
    processor = AutoProcessor.from_pretrained(
        model_source, revision=args.model_revision, local_files_only=args.model_root is not None, max_pixels=args.max_pixels,
    )
    peft = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", target_modules=target_modules, task_type="CAUSAL_LM",
    )
    training = SFTConfig(
        output_dir=str(output_dir), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_length=None,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        completion_only_loss=True, eval_strategy="epoch", save_strategy="epoch", logging_steps=1,
        load_best_model_at_end=args.max_steps < 1, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=2,
        report_to="none", bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        remove_unused_columns=False, seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model, args=training, train_dataset=Dataset.from_list(materialize(train_rows)),
        eval_dataset=Dataset.from_list(materialize(eval_rows)), processing_class=processor, peft_config=peft,
    )
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    wall_clock = time.perf_counter() - started
    for record in trainer.state.log_history:
        for key in ("loss", "eval_loss", "grad_norm"):
            if key in record and not math.isfinite(float(record[key])):
                raise RuntimeError(f"non-finite {key} at step {record.get('step')}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    log_path = output_dir / "train_log.json"
    write_json(log_path, {"wall_clock_seconds": wall_clock, "trainable_parameters": trainable, "log_history": trainer.state.log_history})
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, {
        "wall_clock_seconds": wall_clock,
        "trainable_parameters": trainable,
        "final_log": trainer.state.log_history[-1] if trainer.state.log_history else {},
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
    })
    key_outputs = [log_path, metrics_path, output_dir / "adapter_config.json", output_dir / "adapter_model.safetensors"]
    finalize_run(
        output_dir, receipt, status="COMPLETED", outputs=key_outputs,
        vram_peak_bytes=int(torch.cuda.max_memory_allocated()), trainable_parameters=trainable,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
