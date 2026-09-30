#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
REV=66285546d2b821cf421d4f5eb2576359d3770cd3
DATA_REV=0633573d0438be1185bddebdf1a2c8f5505f7b2a
RUNS=artifacts/v2/runs
SPLIT=artifacts/v2/splits/paper_final_test_receipt.json
mkdir -p "$RUNS" outputs/v2/logs artifacts/v2/manifests

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

test ! -e "$SPLIT"
method_commit=$(git rev-parse HEAD)
"$PY" scripts/freeze_v2_final_split.py --phase-json artifacts/data_audit/source/meta/phase.json --development-split artifacts/splits/split_manifest.json --method-selection artifacts/v2/method_selection.json --dataset-revision "$DATA_REV" --method-commit "$method_commit" --seed 20260930 --output "$SPLIT"
"$PY" scripts/build_reboot_manifest.py --phase-json artifacts/data_audit/source/meta/phase.json --split-manifest "$SPLIT" --frames 4 --span-seconds 2 --output artifacts/v2/manifests/final_sparse.jsonl --audit artifacts/v2/manifests/final_sparse_audit.json >outputs/v2/logs/final-manifest.log

camera=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/method_selection.json"))["camera"])')
method=$("$PY" -c 'import json; print(json.load(open("artifacts/v2/method_selection.json"))["selected"])')
case "$camera" in
  C0) cameras=(observation.images.cam_high observation.images.cam_low) ;;
  C1) cameras=(observation.images.cam_left_wrist observation.images.cam_right_wrist) ;;
  C2) cameras=(observation.images.cam_high observation.images.cam_left_wrist) ;;
  *) exit 1 ;;
esac
sparse=/root/autodl-tmp/prepared/v2/final-sparse-$camera
run final-sparse-materialize "$PY" scripts/prepare_reboot_vlm_dataset.py --manifest artifacts/v2/manifests/final_sparse.jsonl --root /root/autodl-tmp/reboot_sample --output-dir "$sparse" --split-receipt "$SPLIT" --cameras "${cameras[@]}" --num-frames 4 --ablation A2
train_data="$sparse"
eval_data="$sparse"
if [[ "$method" == dense-visual ]]; then
  "$PY" scripts/build_reboot_manifest.py --phase-json artifacts/data_audit/source/meta/phase.json --split-manifest "$SPLIT" --frames 4 --span-seconds 2 --dense --output artifacts/v2/manifests/final_dense.jsonl --audit artifacts/v2/manifests/final_dense_audit.json >outputs/v2/logs/final-dense-manifest.log
  train_data=/root/autodl-tmp/prepared/v2/final-dense-$camera
  run final-dense-materialize "$PY" scripts/prepare_reboot_vlm_dataset.py --manifest artifacts/v2/manifests/final_dense.jsonl --root /root/autodl-tmp/reboot_sample --output-dir "$train_data" --split-receipt "$SPLIT" --cameras "${cameras[@]}" --num-frames 4 --ablation A2
elif [[ "$method" == sparse-trace ]]; then
  train_data=/root/autodl-tmp/prepared/v2/final-trace-$camera
  eval_data="$train_data"
  run final-trace-materialize "$PY" scripts/prepare_reboot_vlm_dataset.py --manifest artifacts/v2/manifests/final_sparse.jsonl --root /root/autodl-tmp/reboot_sample --output-dir "$train_data" --split-receipt "$SPLIT" --cameras "${cameras[@]}" --num-frames 4 --ablation A3
elif [[ "$method" != sparse-visual ]]; then
  echo "invalid method $method" >&2; exit 1
fi

steps=$("$PY" -c 'from pathlib import Path; n=sum(bool(x.strip()) for x in Path(__import__("sys").argv[1]).read_text().splitlines()); print(max(1,2*n//8))' "$sparse/train.jsonl")
"$PY" -c 'import json,sys; from pathlib import Path; Path("artifacts/v2/final_budget.json").write_text(json.dumps({"train_update_budget":int(sys.argv[1]),"rule":"floor(2*sparse_train_rows/gradient_accumulation_8), shared across selected input and full-schema control","method":sys.argv[2],"camera":sys.argv[3]},indent=2)+"\n")' "$steps" "$method" "$camera"

for seed in 42 43 44; do
  name="final-sft-state-seed$seed"
  run "$name" "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$train_data" --dataset-revision "$DATA_REV" --task-schema state-only --epochs 2 --max-steps "$steps" --seed "$seed" --output-dir "$RUNS/$name"
  run "$name-val" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$name" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-val"
done

allow_multitask=$("$PY" -c 'import json; print(int(json.load(open("artifacts/v2/method_selection.json"))["multitask_isolation_allowed"]))')
if [[ "$allow_multitask" == 1 ]]; then
  run final-base-full-val "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/final-base-full-val"
  name=final-sft-full-seed42
  run "$name" "$PY" scripts/train_sft.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$train_data" --dataset-revision "$DATA_REV" --task-schema full --epochs 2 --max-steps "$steps" --seed 42 --output-dir "$RUNS/$name"
  run "$name-val" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/$name" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split val --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name-val"
  "$PY" scripts/decide_v2_rl_gate.py --base-predictions "$RUNS/final-base-full-val/predictions.jsonl" --sft-predictions "$RUNS/$name-val/predictions.jsonl" --output artifacts/v2/final_rl_gate_decision.json
fi

# Fixed methods and checkpoints exist before any new frozen-test prediction is generated.
run final-base-state "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/final-base-state"
for seed in 42 43 44; do
  name="final-sft-state-seed$seed-test"
  run "$name" "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/final-sft-state-seed$seed" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema state-only --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/$name"
done
if [[ "$allow_multitask" == 1 ]]; then
  run final-base-full-test "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/final-base-full-test"
  run final-sft-full-seed42-test "$PY" scripts/eval_base.py --model-root "$MODEL" --model-revision "$REV" --adapter "$RUNS/final-sft-full-seed42" --dataset-root "$eval_data" --dataset-revision "$DATA_REV" --task-schema full --semantic-diagnostic --split test --num-frames 4 --cameras "${cameras[@]}" --load-in-4bit --output-dir "$RUNS/final-sft-full-seed42-test"
fi

mkdir -p artifacts/v2/paired
for seed in 42 43 44; do
  "$PY" scripts/paired_compare.py --first "$RUNS/final-base-state/predictions.jsonl" --second "$RUNS/final-sft-state-seed$seed-test/predictions.jsonl" --task-schema state-only --output "artifacts/v2/paired/final-base-to-state-seed$seed.json" >"outputs/v2/logs/final-paired-state-seed$seed.log"
done
if [[ "$allow_multitask" == 1 ]]; then
  "$PY" scripts/paired_compare.py --first "$RUNS/final-base-full-test/predictions.jsonl" --second "$RUNS/final-sft-full-seed42-test/predictions.jsonl" --task-schema full --output artifacts/v2/paired/final-base-to-full-seed42.json >outputs/v2/logs/final-paired-full.log
fi
