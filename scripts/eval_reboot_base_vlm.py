from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reboot_recovery.metrics import evaluate_predictions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_images(base: Path, rel_paths: list[str]):
    from PIL import Image
    images = []
    for rel in rel_paths:
        with Image.open(base / rel) as image:
            images.append(image.convert("RGB").copy())
    return images


def _generate_one(model, processor, row: dict[str, Any], image_base: Path, max_new_tokens: int) -> tuple[str, float]:
    import torch

    images = _load_images(image_base, row["images"])
    rendered = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    latency = time.perf_counter() - start
    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = generated[:, prompt_len:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return text.strip(), latency


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an untouched or adapted VLM on held-out REBOOT execution windows")
    parser.add_argument("--data", type=Path, required=True, help="Prepared val/test JSONL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reboot_base_predictions.jsonl")
    parser.add_argument("--metrics", type=Path, default=ROOT / "outputs" / "reboot_base_metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-pixels", type=int, default=200704, help="Per-image processor cap; default 256*28*28")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    image_base = args.data.parent
    missing = [str(image_base / p) for row in rows for p in row["images"] if not (image_base / p).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} prepared images; first={missing[:3]}")
    if args.dry_run:
        print(json.dumps({"rows": len(rows), "model": args.model, "images": sum(len(r["images"]) for r in rows)}, indent=2))
        return 0

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("GPU evaluation is required for this VLM pilot.")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        device_map="auto",
        quantization_config=quantization,
    )
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        output, latency = _generate_one(model, processor, row, image_base, args.max_new_tokens)
        predictions.append({
            "id": row["id"],
            "reference": row["reference"],
            "output": output,
            "latency_seconds": latency,
        })
        print(f"[{idx}/{len(rows)}] {row['id']} {latency:.2f}s {output[:160]!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = evaluate_predictions(predictions)
    metrics["mean_latency_seconds"] = (
        sum(float(x["latency_seconds"]) for x in predictions) / len(predictions) if predictions else 0.0
    )
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
