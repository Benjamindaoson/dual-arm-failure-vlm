from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _with_images(rows: list[dict[str, Any]], base: Path) -> list[dict[str, Any]]:
    from PIL import Image
    materialized: list[dict[str, Any]] = []
    for row in rows:
        images = []
        for rel in row["images"]:
            with Image.open(base / rel) as image:
                images.append(image.convert("RGB").copy())
        materialized.append({
            "images": images,
            "prompt": row["prompt"],
            "completion": row["completion"],
        })
    return materialized


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT for the REBOOT execution critic")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reboot_sft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_rows = _read_jsonl(args.train)
    eval_rows = _read_jsonl(args.eval)
    for path, rows in ((args.train, train_rows), (args.eval, eval_rows)):
        missing = [str(path.parent / p) for r in rows for p in r["images"] if not (path.parent / p).exists()]
        if missing:
            raise SystemExit(f"missing prepared images under {path.parent}; first={missing[:3]}")
    if args.dry_run:
        print(json.dumps({
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_images": sum(len(x["images"]) for x in train_rows),
            "eval_images": sum(len(x["images"]) for x in eval_rows),
            "model": args.model,
        }, indent=2))
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    train_ds = Dataset.from_list(_with_images(train_rows, args.train.parent))
    eval_ds = Dataset.from_list(_with_images(eval_rows, args.eval.parent))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization,
    )
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=None,
        completion_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
