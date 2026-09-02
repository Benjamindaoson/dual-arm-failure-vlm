"""Run the course-style visual GRPO/GSPO LoRA training stage on one GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "run_name", "hardware_profile", "base_model", "model_parameter_billion",
    "dataset_id", "dataset_split", "output_dir", "precision", "use_lora",
    "lora_r", "max_seq_length", "num_generations", "learning_rate",
    "per_device_train_batch_size", "gradient_accumulation_steps", "rewards",
}


def load_training_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or REQUIRED - config.keys():
        raise ValueError("missing required chart GSPO training config fields")
    if config["hardware_profile"] == "single_v100":
        if config["precision"] != "fp16" or float(config["model_parameter_billion"]) > 3:
            raise ValueError("single_v100 requires fp16 and a model_parameter_billion value no larger than 3")
    if int(config["num_generations"]) < 2:
        raise ValueError("GSPO requires at least two generations per prompt")
    return config


def _completion_text(completion: object) -> str:
    if isinstance(completion, list) and completion:
        completion = completion[-1]
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def run_training(config: Mapping[str, object]) -> None:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is required for chart GSPO training")
    subprocess.run(["nvidia-smi"], check=True)
    try:
        import torch
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastVisionModel
    except ImportError as error:
        raise RuntimeError("install GPU dependencies: torch datasets transformers trl peft accelerate bitsandbytes unsloth") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after nvidia-smi preflight")
    sys.path.insert(0, str(ROOT / "upgraded_implementation" / "src"))
    from multimodal_chart_gspo.rewards import score_output

    dataset = load_dataset(str(config["dataset_id"]), split=str(config["dataset_split"]))
    image_key = next((key for key in ("decoded_image", "image") if key in dataset.column_names), None)
    question_key = next((key for key in ("question", "query") if key in dataset.column_names), None)
    if image_key is None or question_key is None or "answer" not in dataset.column_names:
        raise ValueError("dataset must provide an image/decoded_image, question/query, and answer column")

    def format_example(row: Mapping[str, object]) -> dict[str, object]:
        text = f"Analyze the chart and answer the question. Return only <answer>...</answer>.\nQuestion: {row[question_key]}"
        return {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}], "image": row[image_key], "answer": str(row["answer"])}

    dataset = dataset.filter(lambda row: row["answer"] is not None).map(format_example, remove_columns=dataset.column_names)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(config["base_model"]), max_seq_length=int(config["max_seq_length"]),
        dtype=torch.float16, load_in_4bit=True, fast_inference=False,
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=int(config["lora_r"]), lora_alpha=int(config["lora_r"]), lora_dropout=0,
        bias="none", random_state=3407, use_rslora=False, loftq_config=None,
    )
    FastVisionModel.for_training(model)

    def correctness_reward(completions: list[object], answer: list[object], **_: object) -> list[float]:
        return [score_output(str(reference), _completion_text(output)).correctness for output, reference in zip(completions, answer)]

    def format_reward(completions: list[object], answer: list[object], **_: object) -> list[float]:
        return [score_output(str(reference), _completion_text(output)).format_reward for output, reference in zip(completions, answer)]

    args = GRPOConfig(
        output_dir=str(config["output_dir"]), run_name=str(config["run_name"]),
        learning_rate=float(config["learning_rate"]), per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]), num_generations=int(config["num_generations"]),
        max_prompt_length=int(config["max_seq_length"]) - 512, max_completion_length=512,
        fp16=True, bf16=False, save_strategy="steps", save_steps=25, logging_steps=1, report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=args, processing_class=tokenizer, reward_funcs=[format_reward, correctness_reward], train_dataset=dataset)
    trainer.train()
    trainer.save_model(str(config["output_dir"]))
    tokenizer.save_pretrained(str(config["output_dir"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a visual chart reasoner with course-style GSPO")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_gspo.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "run_name": config["run_name"], "profile": config["hardware_profile"], "dataset_id": config["dataset_id"]}, ensure_ascii=False))
        return 0
    run_training(config)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
