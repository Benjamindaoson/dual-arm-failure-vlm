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

## CPU-first data audit

After downloading the dataset metadata:

```bash
python scripts/build_reboot_manifest.py \
  --phase-json /path/to/meta/phase.json \
  --output outputs/reboot_manifest.jsonl \
  --audit outputs/reboot_manifest_audit.json

python -m unittest discover -s tests -v
```

The builder:

- splits at episode level to prevent frame leakage;
- creates balanced nominal/failure/recovery temporal anchors;
- samples frame indices deterministically;
- quarantines inconsistent annotations rather than silently repairing them;
- emits class counts and an annotation audit.

## Reward design

`src/reboot_recovery/rewards.py` implements a structured task reward for future RLVR experiments. JSON validity is a **gate, not a bonus**. The task reward comes from phase/state/failure-mode correctness so the model cannot improve its score merely by learning formatting.

## Experiment plan

Read [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md). The next executable stages are:

1. run annotation/data audit on all 60 pilot episodes;
2. build Base VLM evaluation with two fixed camera views and short temporal windows;
3. run SFT only if Base has measurable headroom;
4. add robot state/action trace as an ablation;
5. run GRPO/GSPO only if structured outcome reward improves beyond SFT.

## Legacy material

The original MathVista/Qwen-VL course reproduction remains in the repository for provenance. It is no longer the main project or evidence source for the robotics experiment.
