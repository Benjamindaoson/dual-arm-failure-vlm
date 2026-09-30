#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
REV=66285546d2b821cf421d4f5eb2576359d3770cd3
DATA_REV=0633573d0438be1185bddebdf1a2c8f5505f7b2a
RUNS=artifacts/v2/runs
GATE=artifacts/v2/rl_gate_decision.json
mkdir -p outputs/v2/logs "$RUNS"

decision=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/rl_gate_decision.json"))["decision"])')
if [[ "$decision" != RUN_RLVR ]]; then
  printf 'Formal RL skipped by validation gate: %s\n' "$decision"
  exit 0
fi

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

camera=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/method_selection.json"))["camera"])')
method=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/method_selection.json"))["selected"])')
case "$camera" in
  C0) cameras=(observation.images.cam_high observation.images.cam_low); sparse=/root/autodl-tmp/prepared/A2; trace=/root/autodl-tmp/prepared/v2/A3-high-low ;;
  C1) cameras=(observation.images.cam_left_wrist observation.images.cam_right_wrist); sparse=/root/autodl-tmp/prepared/v2/C1; trace=/root/autodl-tmp/prepared/v2/A3-C1 ;;
  C2) cameras=(observation.images.cam_high observation.images.cam_left_wrist); sparse=/root/autodl-tmp/prepared/v2/C2; trace=/root/autodl-tmp/prepared/v2/A3-C2 ;;
  *) exit 1 ;;
esac
case "$method" in
  sparse-visual) train_data="$sparse"; eval_data="$sparse" ;;
  dense-visual) train_data="/root/autodl-tmp/prepared/v2/dense-$camera"; eval_data="$sparse" ;;
  sparse-trace) train_data="$trace"; eval_data="$trace" ;;
  *) exit 1 ;;
esac

for profile in additive gated; do
  config=configs/reboot_grpo_v2_additive.json
  if [[ "$profile" == gated ]]; then config=configs/reboot_grpo_v2.json; fi
  for steps in 5 50 100; do
    name="rl-$profile-$steps"
    run "$name" "$PY" scripts/train_rlvr.py --config "$config" --gate-decision "$GATE" --dataset-root "$train_data" --model-root "$MODEL" --sft-checkpoint "$RUNS/multitask-sft" --dataset-revision "$DATA_REV" --model-revision "$REV" --max-steps "$steps" --seed 42 --output-dir "$RUNS/$name"
    "$PY" scripts/check_rl_stability.py --run "$RUNS/$name" --steps "$steps" >"outputs/v2/logs/$name.stability.json"
    if [[ "$steps" == 5 ]]; then
      run "$name-reload" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$name" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split val --limit 1 --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-reload"
    else
      for split in val test; do
        suffix=val
        if [[ "$split" == test ]]; then suffix=v1-diagnostic; fi
        run "$name-$suffix" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$name" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split "$split" --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-$suffix"
      done
    fi
  done
done
