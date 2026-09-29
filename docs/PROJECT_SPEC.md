# REBOOT precision-assembly recovery: frozen pilot spec

## Why this dataset

The project starts from REBOOT rather than from a preferred algorithm. REBOOT contains real bimanual precision-assembly trajectories with five shared phases, deliberately induced failures, recovery onset, categorical failure modes, multi-view observations, robot state/action traces, and human recovery descriptions.

The first pilot uses `REBOOT26/sample_recovery-demonstration` (16 mm cylinder install, 60 recovery episodes) because it is small enough to audit end-to-end before scaling to the 18-task suite.

## Question

Can a general-purpose VLM infer the execution phase and recognize when a precision-assembly trajectory has moved from nominal execution into failure and recovery, using a short multi-view temporal window?

This is intentionally narrower than claiming end-to-end robot control. The VLM is a **failure-aware execution critic**. It can later provide a structured state to a recovery policy or robot skill planner.

## Ground-truth targets

Targets come only from `meta/phase.json`:

- `phase`: one of five REBOOT phases.
- `state`: `nominal`, `failure`, or `recovery`, derived from `induced_at_frame` and `recovery_started_at_frame`.
- `failure_mode`: the dataset annotation for non-nominal windows; `none` for nominal windows.
- `recovery_description`: human annotation used as optional SFT text, **not** as a verifiable RL reward in the pilot.

No pseudo-labels are silently introduced.

## Data contract and leakage control

Every model sample is a temporal window from exactly one episode. Dataset splits are made at the **episode level**, never at the frame/window level. All windows from an episode remain in one split.

Annotation-inconsistent episodes are quarantined instead of repaired automatically. This is required because the public annotation file contains edge cases that must be audited before training.

Pilot visual input: four ordered timestamps spanning two seconds from `cam_high` and `cam_low` (eight images). Wrist cameras and robot state/action traces are ablations, not mandatory inputs.

## Experiment gates

1. **Base gate** — Evaluate an untouched VLM. If base performance is already saturated, do not train.
2. **SFT gate** — Train only if base has a measurable failure. Compare against base on held-out episodes.
3. **Trace gate** — Compare single frame vs temporal window; then add compact state/action trace. Keep trace only if it adds held-out value.
4. **RL gate** — Use RLVR only if SFT still leaves meaningful errors on structured targets. Reward phase/state/failure-mode correctness, not prose style.
5. **GSPO gate** — Compare GRPO and sequence-level GSPO only after a real RL signal exists. Do not name the project after GSPO unless the experiment earns that conclusion.

## Minimum metrics

- execution-state macro F1 and per-class recall;
- failure recall (primary safety metric);
- phase macro F1;
- failure-mode macro F1 on non-nominal windows;
- JSON valid rate reported separately, never folded into capability score;
- per-episode bootstrap confidence interval;
- failure timing slices: pre-onset / early failure / late failure / recovery.

## Scale-up after the 60-episode pilot

Expand across connectors and tolerances, then hold out an object/mechanism family to test whether the critic learned transferable recovery state rather than memorizing one task. Only after cross-task evidence should a recovery-skill planner or VLA policy update be added.
