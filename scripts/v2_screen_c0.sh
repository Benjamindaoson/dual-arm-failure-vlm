#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA=/root/autodl-tmp/prepared/A2
MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
REV=66285546d2b821cf421d4f5eb2576359d3770cd3
DATA_REV=0633573d0438be1185bddebdf1a2c8f5505f7b2a
RUNS=artifacts/v2/runs
mkdir -p "$RUNS" outputs/v2/logs

run() {
  local name="$1"
  shift
  "$@" >"outputs/v2/logs/$name.stdout.log" 2>"outputs/v2/logs/$name.stderr.log" || {
    echo "$name FAILED; see outputs/v2/logs/$name.stderr.log" >&2
    return 1
  }
  cp "outputs/v2/logs/$name.stdout.log" "$RUNS/$name/stdout.log"
  cp "outputs/v2/logs/$name.stderr.log" "$RUNS/$name/stderr.log"
}

run c0-base-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$DATA" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras observation.images.cam_high observation.images.cam_low --load-in-4bit --output-dir "$RUNS/c0-base-val"
run c0-sft-smoke "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$DATA" --dataset-revision "$DATA_REV" --task-schema state-only --max-steps 5 --output-dir "$RUNS/c0-sft-smoke"
run c0-sft "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$DATA" --dataset-revision "$DATA_REV" --task-schema state-only --epochs 2 --output-dir "$RUNS/c0-sft"
run c0-sft-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/c0-sft" --dataset-root "$DATA" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras observation.images.cam_high observation.images.cam_low --load-in-4bit --output-dir "$RUNS/c0-sft-val"
run c0-sft-v1-diagnostic "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/c0-sft" --dataset-root "$DATA" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras observation.images.cam_high observation.images.cam_low --load-in-4bit --output-dir "$RUNS/c0-sft-v1-diagnostic"
printf 'C0 screening completed at %s\n' "$(date -u +%FT%TZ)"
