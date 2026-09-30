from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.evidence import finalize_run, write_run_start, write_json
from reboot_recovery.metrics import diagnose_predictions, evaluate_predictions, evaluate_state_predictions, normalize_semantic_output
from reboot_recovery.prompts import to_state_only_record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Base or adapted VLM on one held-out REBOOT split")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None, help="Optional PEFT adapter trained from the selected base model")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--task-schema", choices=["full", "state-only"], default="full")
    parser.add_argument("--semantic-diagnostic", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--cameras", nargs="+", default=["observation.images.cam_high", "observation.images.cam_low"])
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_images(base: Path, relative_paths: list[str]):
    from PIL import Image

    images = []
    for relative in relative_paths:
        with Image.open(base / relative) as image:
            images.append(image.convert("RGB").copy())
    return images


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = args.dataset_root / f"{args.split}.jsonl"
    if not data_path.is_file():
        raise SystemExit(f"prepared split not found: {data_path}")
    rows = _read_jsonl(data_path)
    if args.limit is not None:
        rows = rows[:args.limit]
    expected_images = args.num_frames * len(args.cameras)
    invalid_counts = [row.get("id", "<missing>") for row in rows if len(row.get("images", [])) != expected_images]
    if invalid_counts:
        raise SystemExit(f"{len(invalid_counts)} rows do not contain {expected_images} images; first={invalid_counts[:3]}")
    missing = [str(args.dataset_root / relative) for row in rows for relative in row["images"] if not (args.dataset_root / relative).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} prepared images; first={missing[:3]}")
    if args.task_schema == "state-only":
        rows = [to_state_only_record(row) for row in rows]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "artifacts" / "runs" / f"base-{args.split}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    model_source = str(args.model_root or args.model)
    if args.adapter is not None and not args.adapter.is_dir():
        raise SystemExit(f"adapter directory not found: {args.adapter}")
    config = {
        "stage": "adapter_eval" if args.adapter else "base_eval",
        "model": model_source,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "model_revision": args.model_revision,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_revision": args.dataset_revision,
        "task_schema": args.task_schema,
        "semantic_diagnostic": args.semantic_diagnostic or args.task_schema == "state-only",
        "split": args.split,
        "num_frames": args.num_frames,
        "cameras": args.cameras,
        "max_pixels": args.max_pixels,
        "limit": args.limit,
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    receipt = write_run_start(
        output_dir, root=ROOT, run_id=output_dir.name, config=config,
        dataset_revision=args.dataset_revision, model_revision=args.model_revision, seed=args.seed,
    )
    if args.dry_run:
        summary = {"status": "DRY_RUN_VERIFIED", "rows": len(rows), "images": sum(len(row["images"]) for row in rows)}
        write_json(output_dir / "dry_run.json", summary)
        finalize_run(output_dir, receipt, status="DRY_RUN_VERIFIED", outputs=[output_dir / "dry_run.json"])
        print(json.dumps(summary, indent=2))
        return 0

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        finalize_run(output_dir, receipt, status="BLOCKED_NO_CUDA")
        raise SystemExit("GPU evaluation is required for Qwen2.5-VL-3B.")
    torch.manual_seed(args.seed)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype
    ) if args.load_in_4bit else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_source, revision=args.model_revision, local_files_only=args.model_root is not None,
        torch_dtype=compute_dtype, device_map="auto", quantization_config=quantization,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
    processor = AutoProcessor.from_pretrained(
        model_source, revision=args.model_revision, local_files_only=args.model_root is not None, max_pixels=args.max_pixels,
    )
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        images = _load_images(args.dataset_root, row["images"])
        rendered = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        latency = time.perf_counter() - started
        new_tokens = generated[:, int(inputs["input_ids"].shape[1]):]
        output = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        predictions.append({
            "id": row["id"],
            "reference": row["reference"],
            "output": output,
            "latency_seconds": latency,
            "split_receipt_sha256": row.get("split_receipt_sha256"),
        })
        print(f"[{index}/{len(rows)}] {row['id']} {latency:.2f}s")

    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    if args.task_schema == "state-only":
        metrics = evaluate_state_predictions(predictions)
        normalized = [{**row, "output": normalize_semantic_output(row["output"])} for row in predictions]
        metrics["semantic_diagnostic"] = evaluate_state_predictions(normalized)
        metrics["outer_fence_removed_count"] = sum(
            row["output"].strip() != item["output"] for row, item in zip(predictions, normalized, strict=True)
        )
    elif args.semantic_diagnostic:
        diagnostic = diagnose_predictions(predictions)
        metrics = diagnostic["protocol"]
        metrics["semantic_diagnostic"] = diagnostic["semantic"]
        metrics["outer_fence_removed_count"] = diagnostic["outer_fence_removed_count"]
    else:
        metrics = evaluate_predictions(predictions)
    if args.task_schema == "state-only" or args.semantic_diagnostic:
        metrics["task_schema"] = args.task_schema
    metrics["mean_latency_seconds"] = sum(row["latency_seconds"] for row in predictions) / len(predictions) if predictions else 0.0
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    vram = int(torch.cuda.max_memory_allocated())
    finalize_run(output_dir, receipt, status="COMPLETED", outputs=[prediction_path, metrics_path], vram_peak_bytes=vram)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
