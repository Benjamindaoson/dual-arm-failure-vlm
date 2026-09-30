#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
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

for variant in C1 C2; do
  data="/root/autodl-tmp/prepared/v2/$variant"
  test -f "$data/dataset_receipt.json"
  if [[ "$variant" == C1 ]]; then
    cameras=(observation.images.cam_left_wrist observation.images.cam_right_wrist)
  else
    cameras=(observation.images.cam_high observation.images.cam_left_wrist)
  fi
  name="${variant,,}"
  run "$name-base-val" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-base-val"
  run "$name-sft-smoke" "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --max-steps 5 --output-dir "$RUNS/$name-sft-smoke"
  run "$name-sft" "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --epochs 2 --output-dir "$RUNS/$name-sft"
  run "$name-sft-val" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$name-sft" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-sft-val"
  run "$name-sft-v1-diagnostic" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$name-sft" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-sft-v1-diagnostic"
done
"$PY" scripts/select_v2_camera.py --runs-root "$RUNS" --output artifacts/v2/camera_selection.json
