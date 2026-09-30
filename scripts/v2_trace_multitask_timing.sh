#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
REV=66285546d2b821cf421d4f5eb2576359d3770cd3
DATA_REV=0633573d0438be1185bddebdf1a2c8f5505f7b2a
RUNS=artifacts/v2/runs
mkdir -p "$RUNS" outputs/v2/logs artifacts/v2/timing

run() {
  local name="$1"
  shift
  "$@" >"outputs/v2/logs/$name.stdout.log" 2>"outputs/v2/logs/$name.stderr.log" || {
    echo "$name FAILED; see outputs/v2/logs/$name.stderr.log" >&2
    return 1
  }
  if [[ -d "$RUNS/$name" ]]; then
    cp "outputs/v2/logs/$name.stdout.log" "$RUNS/$name/stdout.log"
    cp "outputs/v2/logs/$name.stderr.log" "$RUNS/$name/stderr.log"
  fi
}

camera=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/camera_selection.json"))["selected"])')
case "$camera" in
  C0) cameras=(observation.images.cam_high observation.images.cam_low); sparse=/root/autodl-tmp/prepared/A2; trace=/root/autodl-tmp/prepared/v2/A3-high-low ;;
  C1) cameras=(observation.images.cam_left_wrist observation.images.cam_right_wrist); sparse=/root/autodl-tmp/prepared/v2/C1; trace=/root/autodl-tmp/prepared/v2/A3-C1 ;;
  C2) cameras=(observation.images.cam_high observation.images.cam_left_wrist); sparse=/root/autodl-tmp/prepared/v2/C2; trace=/root/autodl-tmp/prepared/v2/A3-C2 ;;
  *) echo "invalid selected camera $camera" >&2; exit 1 ;;
esac

if [[ ! -f "$trace/dataset_receipt.json" ]]; then
  run trace-materialize "$PY" scripts/prepare_reboot_vlm_dataset.py --manifest artifacts/manifests/reboot_windows.jsonl --root /root/autodl-tmp/reboot_sample --output-dir "$trace" --cameras "${cameras[@]}" --num-frames 4 --ablation A3
fi
run trace-sft-smoke "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$trace" --dataset-revision "$DATA_REV" --task-schema state-only --max-steps 5 --output-dir "$RUNS/trace-sft-smoke"
run trace-sft "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$trace" --dataset-revision "$DATA_REV" --task-schema state-only --epochs 2 --output-dir "$RUNS/trace-sft"
run trace-sft-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/trace-sft" --dataset-root "$trace" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/trace-sft-val"
run trace-sft-v1-diagnostic "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/trace-sft" --dataset-root "$trace" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/trace-sft-v1-diagnostic"

"$PY" scripts/select_v2_method.py --runs-root "$RUNS" --camera-selection artifacts/v2/camera_selection.json --output artifacts/v2/method_selection.json
method=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/method_selection.json"))["selected"])')
case "$method" in
  sparse-visual) train_data="$sparse"; eval_data="$sparse"; state_run="${camera,,}-sft" ;;
  dense-visual) train_data="/root/autodl-tmp/prepared/v2/dense-$camera"; eval_data="$sparse"; state_run=dense-sft ;;
  sparse-trace) train_data="$trace"; eval_data="$trace"; state_run=trace-sft ;;
  *) echo "invalid selected method $method" >&2; exit 1 ;;
esac
steps=$("$PY" -c 'import json,sys; print(json.load(open("artifacts/v2/runs/"+sys.argv[1]+"/metrics.json"))["global_step"])' "$state_run")
allow_multitask=$("$PY" -c 'import json; print(int(json.load(open("artifacts/v2/method_selection.json"))["multitask_isolation_allowed"]))')
if [[ "$allow_multitask" == 1 ]]; then
  run multitask-base-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/multitask-base-val"
  run multitask-sft-smoke "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$train_data" --dataset-revision "$DATA_REV" --task-schema full --max-steps 5 --output-dir "$RUNS/multitask-sft-smoke"
  run multitask-sft "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$train_data" --dataset-revision "$DATA_REV" --task-schema full --max-steps "$steps" --output-dir "$RUNS/multitask-sft"
  run multitask-sft-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/multitask-sft" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/multitask-sft-val"
  run multitask-sft-v1-diagnostic "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/multitask-sft" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/multitask-sft-v1-diagnostic"
  "$PY" scripts/decide_v2_rl_gate.py --base-predictions "$RUNS/multitask-base-val/predictions.jsonl" --sft-predictions "$RUNS/multitask-sft-val/predictions.jsonl" --output artifacts/v2/rl_gate_decision.json
else
  "$PY" -c 'import json; from pathlib import Path; m=json.load(open("artifacts/v2/method_selection.json")); Path("artifacts/v2/rl_gate_decision.json").write_text(json.dumps({"decision":"REVISIT_REPRESENTATION_OR_SUPERVISION","reason":"state_only_failure_signal_not_detected_in_two_validation_episodes","selection_sha256":__import__("hashlib").sha256(Path("artifacts/v2/method_selection.json").read_bytes()).hexdigest()},indent=2)+"\n")'
fi

timing_data="/root/autodl-tmp/prepared/v2/timing-$method-$camera"
timing_ablation=A2
if [[ "$method" == sparse-trace ]]; then timing_ablation=A3; fi
run timing-materialize "$PY" scripts/prepare_reboot_vlm_dataset.py --manifest artifacts/eval/failure_timing_manifest.jsonl --root /root/autodl-tmp/reboot_sample --output-dir "$timing_data" --cameras "${cameras[@]}" --num-frames 4 --ablation "$timing_ablation"
run timing-state-only "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$state_run" --dataset-root "$timing_data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/timing-state-only"
for event in failure_onset recovery_onset; do
  run "timing-$event" "$PY" scripts/evaluate_failure_timing.py --predictions "$RUNS/timing-state-only/predictions.jsonl" --task-schema state-only --semantic-diagnostic --event-kind "$event" --consecutive 2 --output-json "artifacts/v2/timing/$event.json" --output-csv "artifacts/v2/timing/$event.csv"
done
