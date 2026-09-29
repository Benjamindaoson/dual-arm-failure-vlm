# REBOOT Precision-Assembly Recovery

**Multimodal post-training for failure-aware robot execution.**

This branch upgrades the original course-style MathVista GSPO experiment into a dataset-first robotics project based on **REBOOT**, a real bimanual precision-assembly benchmark with annotated failure and recovery trajectories.

The first deliverable is deliberately not an end-to-end robot controller. It is an auditable multimodal **execution critic** that predicts:

```text
phase        = Align(pick) | Engage(pick) | Transport | Align(place) | Engage(place)
state        = nominal | failure | recovery
failure_mode = dataset annotation | none
```

from a short ordered visual window and the task instruction. Training decisions are gated by held-out evidence: **Base → SFT → RLVR → GRPO/GSPO only when each previous stage leaves a real, measurable problem.**

## Verified pilot data contract

Pilot dataset: `REBOOT26/sample_recovery-demonstration`.

Public metadata declares 60 episodes, 53,886 frames, 30 FPS, 14-D robot state, 14-D action, and four 640×480 RGB cameras. `meta/phase.json` supplies the five phase boundaries per episode plus `originating_phase`, `failure_mode`, `induced_at_frame`, `recovery_started_at_frame`, and a human recovery description.

The pilot uses the annotations as ground truth rather than inventing labels from video.

## Run the pilot

Start with metadata and annotation audit; no GPU is needed:

```bash
python -m pip install huggingface_hub
python scripts/fetch_reboot_pilot.py
python scripts/build_reboot_manifest.py \
  --phase-json data/reboot_sample/meta/phase.json \
  --output outputs/reboot_manifest.jsonl \
  --audit outputs/reboot_manifest_audit.json
python scripts/prepare_reboot_vlm_dataset.py \
  --manifest outputs/reboot_manifest.jsonl \
  --dry-run
python -m unittest discover -s tests -v
```

Then download the 2.84 GB pilot and materialize only the temporal windows used by the experiment:

```bash
python scripts/fetch_reboot_pilot.py --full
python -m pip install -r requirements-reboot.txt
python scripts/prepare_reboot_vlm_dataset.py \
  --manifest outputs/reboot_manifest.jsonl \
  --root data/reboot_sample \
  --output-dir prepared/reboot_vlm
```

Run the untouched VLM on held-out episodes:

```bash
python scripts/eval_reboot_base_vlm.py \
  --data prepared/reboot_vlm/test.jsonl \
  --load-in-4bit \
  --output outputs/reboot_base_predictions.jsonl \
  --metrics outputs/reboot_base_metrics.json

python scripts/decide_reboot_stage.py \
  --base outputs/reboot_base_metrics.json
```

Only when the Base gate says `RUN_SFT`:

```bash
python scripts/train_reboot_sft.py \
  --train prepared/reboot_vlm/train.jsonl \
  --eval prepared/reboot_vlm/val.jsonl \
  --output-dir outputs/reboot_sft
```

Re-run the same evaluator with `--model outputs/reboot_sft`, save SFT metrics, then call `decide_reboot_stage.py --base ... --sft ...`. RL is intentionally not the default next step.

## Data integrity

The builder:

- splits at episode level to prevent frame leakage;
- creates nominal/failure/recovery temporal anchors from the published annotations;
- samples frame indices deterministically;
- quarantines inconsistent annotations rather than silently repairing them;
- emits class counts and an annotation audit.

## Reward design

`src/reboot_recovery/rewards.py` implements a structured task reward for future RLVR experiments. JSON validity is a **gate, not a bonus**. The task reward comes from phase/state/failure-mode correctness so the model cannot improve its score merely by learning formatting.

## Experiment plan

Read [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md). The evidence gates are:

1. Base VLM on held-out episodes.
2. SFT only if Base has measurable headroom.
3. Single frame vs temporal window; robot state/action trace only as an ablation.
4. RLVR only if SFT improves but leaves meaningful outcome errors.
5. GRPO vs sequence-level GSPO only after a real RL signal exists.

## Legacy material

The original MathVista/Qwen-VL course reproduction remains in the repository for provenance. It is no longer the main project or evidence source for the robotics experiment.
