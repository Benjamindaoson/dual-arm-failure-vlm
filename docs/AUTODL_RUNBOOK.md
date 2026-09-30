# AutoDL offline runbook

This runbook keeps datasets, model weights, caches, and checkpoints on `/root/autodl-tmp`. It never requires a token in source control.

## 1. Transfer and bootstrap

Create the bundle on a connected machine:

```bash
python scripts/package_autodl_bundle.py --output dist/reboot-autodl-bundle.zip
```

Upload the ZIP, the pinned REBOOT dataset, Qwen model snapshot, and an optional wheelhouse to `/root/autodl-tmp`. On AutoDL:

```bash
cd /root/autodl-tmp
unzip reboot-autodl-bundle.zip -d multimodal-chart-gspo
cd multimodal-chart-gspo
bash scripts/bootstrap_autodl.sh
source scripts/autodl_env.sh
# Fully offline dependency installation:
bash scripts/bootstrap_autodl.sh --wheelhouse /root/autodl-tmp/wheels
source scripts/autodl_env.sh
```

The bootstrap creates the project-local `.venv`. Source the committed environment file in every new shell so cache placement persists:

```bash
source scripts/autodl_env.sh
HF_HOME=/root/autodl-tmp/hf
HF_HUB_CACHE=/root/autodl-tmp/hf/hub
TORCH_HOME=/root/autodl-tmp/torch
```

## 2. Verify local assets

```bash
.venv/bin/python scripts/verify_local_dataset.py \
  --dataset-root /root/autodl-tmp/reboot_sample \
  --require-full \
  --output artifacts/data_audit/local_dataset_verification.json

test -f /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/config.json
nvidia-smi
```

Stop if dataset verification fails, the model snapshot is incomplete, or CUDA is unavailable.

## 3. Materialize A0–A3 from the frozen split

Run `scripts/build_reboot_manifest.py` against the pinned `meta/phase.json`, then materialize each ablation with the same episode manifest:

```bash
for ABLATION in A0 A1 A2 A3; do
  .venv/bin/python scripts/prepare_reboot_vlm_dataset.py \
    --manifest artifacts/manifests/reboot_windows.jsonl \
    --root /root/autodl-tmp/reboot_sample \
    --output-dir "/root/autodl-tmp/prepared/${ABLATION}" \
    --num-frames 4 --ablation "$ABLATION"
done
```

`A3` uses a compact Trace-Text summary. A learned numeric projector is intentionally deferred until Trace-Text shows held-out value.

## 4. Base evaluation

Run all ablations on the same test episodes:

```bash
for ABLATION in A0 A1 A2 A3; do
  CAMERAS=(observation.images.cam_high observation.images.cam_low)
  FRAMES=4
  [[ "$ABLATION" == A0 ]] && CAMERAS=(observation.images.cam_high) && FRAMES=1
  [[ "$ABLATION" == A1 ]] && CAMERAS=(observation.images.cam_high)
  .venv/bin/python scripts/eval_base.py \
    --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --dataset-root "/root/autodl-tmp/prepared/${ABLATION}" \
    --dataset-revision 0633573d0438be1185bddebdf1a2c8f5505f7b2a \
    --split test --num-frames "$FRAMES" --cameras "${CAMERAS[@]}" --max-pixels 100352 \
    --load-in-4bit --output-dir "artifacts/runs/base-${ABLATION}"
done

.venv/bin/python scripts/compare_input_ablations.py \
  --a0 artifacts/runs/base-A0/predictions.jsonl \
  --a1 artifacts/runs/base-A1/predictions.jsonl \
  --a2 artifacts/runs/base-A2/predictions.jsonl \
  --a3 artifacts/runs/base-A3/predictions.jsonl

.venv/bin/python scripts/decide_reboot_stage.py \
  --base artifacts/runs/base-A2/metrics.json \
  --output artifacts/decisions/base.json
```

## 5. Gated SFT and RLVR

Run SFT only if `artifacts/decisions/base.json` says `RUN_SFT`:

```bash
.venv/bin/python scripts/train_sft.py \
  --dataset-root /root/autodl-tmp/prepared/A2 \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --dataset-revision 0633573d0438be1185bddebdf1a2c8f5505f7b2a \
  --variant SFT-Language --max-steps 5 --output-dir artifacts/runs/sft-smoke

.venv/bin/python scripts/train_sft.py \
  --dataset-root /root/autodl-tmp/prepared/A2 \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --dataset-revision 0633573d0438be1185bddebdf1a2c8f5505f7b2a \
  --variant SFT-Language --output-dir artifacts/runs/sft-main
```

Evaluate the adapter against the same Base model and test split, then pass Base and SFT metrics to `scripts/decide_reboot_stage.py`:

```bash
.venv/bin/python scripts/eval_base.py \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter artifacts/runs/sft-main \
  --dataset-root /root/autodl-tmp/prepared/A2 \
  --split test --num-frames 4 \
  --cameras observation.images.cam_high observation.images.cam_low \
  --load-in-4bit --output-dir artifacts/runs/sft-language-A2

.venv/bin/python scripts/decide_reboot_stage.py \
  --base artifacts/runs/base-A2/metrics.json \
  --sft artifacts/runs/sft-language-A2/metrics.json \
  --output artifacts/decisions/sft.json
```

Run GRPO and GSPO only if the decision is `RUN_RLVR`:

```bash
.venv/bin/python scripts/train_rlvr.py \
  --config configs/reboot_grpo.json --gate-decision artifacts/decisions/sft.json \
  --dataset-root /root/autodl-tmp/prepared/A2 \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --sft-checkpoint artifacts/runs/sft-main \
  --output-dir artifacts/runs/grpo

.venv/bin/python scripts/train_rlvr.py \
  --config configs/reboot_gspo.json --gate-decision artifacts/decisions/sft.json \
  --dataset-root /root/autodl-tmp/prepared/A2 \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --sft-checkpoint artifacts/runs/sft-main \
  --output-dir artifacts/runs/gspo
```

TRL contract: `>=0.29,<0.30`. GRPO uses `importance_sampling_level="token"`; GSPO uses `"sequence"`; both use `loss_type="grpo"`. The combination `sequence + dr_grpo` is rejected as not paper GSPO.

## 6. Failure timing and trace ablation

```bash
.venv/bin/python scripts/prepare_reboot_vlm_dataset.py \
  --manifest artifacts/eval/failure_timing_manifest.jsonl \
  --root /root/autodl-tmp/reboot_sample \
  --output-dir /root/autodl-tmp/prepared/timing-A2 \
  --num-frames 4 --ablation A2

.venv/bin/python scripts/eval_base.py \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --dataset-root /root/autodl-tmp/prepared/timing-A2 --split test \
  --num-frames 4 --cameras observation.images.cam_high observation.images.cam_low \
  --load-in-4bit --output-dir artifacts/runs/timing-base

.venv/bin/python scripts/evaluate_failure_timing.py \
  --predictions artifacts/runs/timing-base/predictions.jsonl \
  --output-json artifacts/eval/failure_timing_base.json \
  --output-csv artifacts/eval/failure_timing_base.csv
```

Repeat the timing evaluation with `--adapter artifacts/runs/sft-main`. Evaluate A3 with the same model as A2 and use `scripts/compare_model_predictions.py`; it rejects mismatched sample IDs and split receipts.

## 7. Resume

```bash
.venv/bin/python scripts/train_sft.py ... \
  --output-dir artifacts/runs/sft-main \
  --resume-from-checkpoint latest
```

Every formal run retains configuration, environment, logs, metrics/predictions, peak VRAM when available, trainable parameter count, and output hashes under its run directory.
