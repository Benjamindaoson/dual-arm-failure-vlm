#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
REV=66285546d2b821cf421d4f5eb2576359d3770cd3
DATA_REV=0633573d0438be1185bddebdf1a2c8f5505f7b2a
RUNS=artifacts/v2/runs
mkdir -p "$RUNS" outputs/v2/logs artifacts/v2/manifests
variant=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/camera_selection.json"))["selected"])')
case "$variant" in
  C0) cameras=(observation.images.cam_high observation.images.cam_low) ;;
  C1) cameras=(observation.images.cam_left_wrist observation.images.cam_right_wrist) ;;
  C2) cameras=(observation.images.cam_high observation.images.cam_left_wrist) ;;
  *) echo "invalid selected camera: $variant" >&2; exit 1 ;;
esac
steps=$("$PY" -c 'import json,sys; print(json.load(open("artifacts/v2/runs/"+sys.argv[1].lower()+"-sft/metrics.json"))["global_step"])' "$variant")
data=/root/autodl-tmp/prepared/v2/dense-$variant
sparse_data=/root/autodl-tmp/prepared/A2
if [[ "$variant" != C0 ]]; then sparse_data="/root/autodl-tmp/prepared/v2/$variant"; fi

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

"$PY" scripts/build_reboot_manifest.py --phase-json artifacts/data_audit/source/meta/phase.json --split-manifest artifacts/splits/split_manifest.json --frames 4 --span-seconds 2 --dense --output artifacts/v2/manifests/dense_windows.jsonl --audit artifacts/v2/manifests/dense_audit.json >outputs/v2/logs/dense-manifest.log 2>&1
"$PY" scripts/prepare_reboot_vlm_dataset.py --manifest artifacts/v2/manifests/dense_windows.jsonl --root /root/autodl-tmp/reboot_sample --output-dir "$data" --cameras "${cameras[@]}" --num-frames 4 --ablation A2 >outputs/v2/logs/dense-materialize.log 2>&1
run dense-sft-smoke "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --max-steps 5 --output-dir "$RUNS/dense-sft-smoke"
run dense-sft "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$data" --dataset-revision "$DATA_REV" --task-schema state-only --epochs 2 --max-steps "$steps" --output-dir "$RUNS/dense-sft"
run dense-sft-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/dense-sft" --dataset-root "$sparse_data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/dense-sft-val"
run dense-sft-v1-diagnostic "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/dense-sft" --dataset-root "$sparse_data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/dense-sft-v1-diagnostic"
