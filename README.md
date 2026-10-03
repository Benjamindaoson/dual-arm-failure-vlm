# REBOOT Precision Assembly Recovery

**Multimodal Post-training for Failure-Aware Robot Execution**

A robot can complete every local skill and still fail the assembly: a connector arrives a few millimetres off-axis, a grasp slips during transport, or insertion jams after apparently correct alignment. This project trains and evaluates a multimodal foundation model as a **failure-aware execution critic** over visual history and robot trace—not as an end-to-end controller and not as a classical CV detector.

The critic answers three grounded questions:

```json
{
  "phase": "Align (place)",
  "state": "failure",
  "failure_mode": "misalignment"
}
```

The project uses public [REBOOT precision-assembly trajectories](https://huggingface.co/datasets/REBOOT26/sample_recovery-demonstration). It is an **independent research project**. It is not Chery internal work, does not use Chery data, and is not an official Georgia Tech research project.

**Measured outcome:** The V1 diagnostic SFT reached State Macro-F1 **0.4012** and strict JSON validity **100%**, but Failure Recall remained **0/18**. The subsequent V2 final refit found that state-only Base had **18/18 semantic failure recall only by also flagging all 18 nominal windows**; three QLoRA SFT seeds fell to **1/18, 7/18, and 7/18** semantic failure recall while reaching 100% strict JSON validity. Full-schema SFT remained at **0/18**. The V2 validation RL gate closed, so **no V2 GRPO/GSPO was run**. See the [full report](docs/EXPERIMENT_REPORT_CN.md), [paper draft](paper/main.tex), and [final split receipt](artifacts/v2/splits/paper_final_test_receipt.json). No checkpoint is qualified as a reliable failure detector.

### Evidence boundary for the V2 result

The V2 final test contains 54 windows from six episodes of one `16mm-cylinder-install` task. Its episodes are disjoint from the **final refit** train and validation sets, but participated in earlier method-development training. This is **internal Tier B evidence, not independent replication or cross-task generalization**. The frozen strict scorer rejects Markdown-fenced JSON; a separately reported semantic diagnostic removes only the outer fence. The Base model's strict 0/18 therefore cannot be used alone to claim SFT improved failure awareness. The [final run receipts and predictions](artifacts/v2/runs/final-base-state/run_receipt.json), [paired seed-43 comparison](artifacts/v2/paired/final-base-to-state-seed43.json), and [evidence verifier output](artifacts/v2/evidence_verification.json) support the reported numbers. Four final adapters are backed up locally outside Git with hashes matched to their receipts.

The historical sections below describe the **V1 pilot**, except where V2 is explicitly named. In particular, V1 GRPO/GSPO runs and timing results are not V2 final-test outcomes.

## Why this problem matters

Precision assembly failures are temporal and contact-dependent. A single image may show the object near the target but not whether the robot drifted, slipped, stalled, or has already begun recovery. REBOOT contributes a shared five-phase task decomposition and failure/recovery annotations across manufacturing-relevant assembly mechanisms. The full benchmark reports 2,160 demonstrations across 18 install/remove tasks; this repository starts with the smaller 60-episode cylinder-install sample so every annotation and split can be audited before scale-up. See the [official REBOOT project page](https://nanayawoa.github.io/REBOOT/).

The first research question is deliberately narrow:

> Does ordered visual history—and then a compact state/action trace—improve held-out phase, failure, and recovery recognition over a single current frame?

No action policy is trained in Phase 1. Recovery-action labels are not invented.

## Audited pilot data

The committed receipt pins dataset revision `0633573d0438be1185bddebdf1a2c8f5505f7b2a` and records source hashes.

| Item | Observed in the pinned sample |
|---|---:|
| Episodes declared / observed | 60 / 60 |
| Frames declared by `meta/info.json` | 53,886 |
| FPS | 30 |
| RGB cameras | 4 |
| Depth fields in this sample snapshot | none |
| Robot state / action shape | 14 / 14 |
| Annotation `episode_index` type | zero-padded string |
| Frame-table `episode_index` dtype | `int64` |
| Valid / quarantined episodes | 53 / 7 |
| Episode split | 42 train / 5 validation / 6 test |

The audit does not smooth over source inconsistencies:

- episodes `16` and `26` have out-of-range one-based `originating_phase` values (`0` and `21`);
- episode `37` has `recovery_started_at_frame < induced_at_frame`;
- `duration_frames` is an inclusive terminal index in this snapshot, so a normal episode has 898 rows for terminal index 897;
- episodes `00`, `15`, `26`, `34`, `37`, and `38` each contain an additional out-of-range frame 898 (899 rows instead of 898);
- all 60 annotation terminal indices sum to 53,820 while the frame tables contain 53,886 rows; the six true extra rows are preserved as anomalies, not repaired;
- `meta/phase.json` uses the plural repository spelling while the actual Hub id is singular.

Those episodes are quarantined before window creation. The pilot remains small; its role is to qualify the task and pipeline, not establish a production-level result.

Evidence:

- [`artifacts/data_audit/DATA_RECEIPT.md`](artifacts/data_audit/DATA_RECEIPT.md)
- [`artifacts/data_audit/reboot_schema.json`](artifacts/data_audit/reboot_schema.json)
- [`artifacts/data_audit/reboot_annotation_audit.json`](artifacts/data_audit/reboot_annotation_audit.json)
- [`artifacts/splits/split_manifest.json`](artifacts/splits/split_manifest.json)
- [`artifacts/splits/split_receipt.json`](artifacts/splits/split_receipt.json)

## Task and input ablations

The output schema is strict: exactly `phase`, `state`, and `failure_mode`. Phase and failure labels come from the pinned annotations. `failure_mode` must be `none` for a nominal window. JSON validity is reported separately and cannot improve the task reward by itself.

All ablations use the same six held-out test episodes:

| Variant | Input |
|---|---|
| A0 | one current frame from `cam_high` |
| A1 | four causal/ordered timestamps from `cam_high` |
| A2 | four timestamps from `cam_high` and `cam_low` (8 images) |
| A3 | A2 plus a compact Trace-Text summary of state delta and latest action |

Trace-MLP is intentionally deferred. It becomes justified only if Trace-Text exposes a held-out signal worth preserving.

## Metrics

The unified evaluator retains every prediction and reports:

- state macro-F1;
- failure recall (primary safety-oriented metric);
- recovery recall;
- phase macro-F1;
- failure-mode macro-F1 on non-nominal windows;
- JSON valid rate, separately;
- per-class precision, recall, F1, and support;
- per-episode bootstrap 95% confidence intervals;
- pre-failure, early/late failure, and early/late recovery slices.

Failure and recovery timing use causal windows at `T-2s`, `T-1s`, `T-0.5s`, `T`, `T+0.5s`, `T+1s`, and `T+2s`. Failure-detection delay is the first timestamp at which the model has predicted failure for `K=2` consecutive windows; pre-onset failure predictions are counted as false alarms.

## Evidence-gated post-training

The repository does not assume that training—or GSPO—helps:

```text
Base evaluation
  ├─ failure recall ≥ 0.90 and state macro-F1 ≥ 0.85 → BASE_SUFFICIENT
  └─ otherwise → RUN_SFT
        ├─ SFT meets targets → SFT_SUFFICIENT
        ├─ SFT improves and leaves outcome errors → RUN_RLVR
        └─ no clean gain → REVISIT_DATA_OR_TASK
```

SFT is 4-bit NF4 QLoRA with rank 16, alpha 32, dropout 0.05, gradient checkpointing, batch size 1, gradient accumulation 8, and BF16 when supported (otherwise FP16). Two dry-runnable variants are available:

- `SFT-Language`: language attention and MLP projections;
- `SFT-VisionLanguage`: all linear vision/projector/language modules, only when memory allows.

RLVR uses phase, state, and applicable failure-mode correctness. Invalid JSON receives zero task reward; valid-but-wrong JSON also receives zero for wrong components. Nominal rows never receive taxonomy credit. Inverse-square-root failure-mode weights address imbalance without adding a format reward.

With TRL `>=0.29,<0.30`:

- GRPO: `importance_sampling_level="token"`, `loss_type="grpo"`;
- GSPO: `importance_sampling_level="sequence"`, `loss_type="grpo"`.

The validator rejects `importance_sampling_level="sequence"` plus `loss_type="dr_grpo"` as “paper GSPO.” The historical V1 comparison used four generations, 96 completion tokens, and 100 update steps for each RL algorithm. V2's stricter validation gate did not permit RL.

## What was executed on GPU in V1

The 4090 D host passed `nvidia-smi`, `/dev/nvidia*`, `torch.cuda.is_available()`, and BF16 checks. A0/A1/A2/A3 were fully materialized from the already downloaded public dataset. Base A0/A1/A2 each had a two-sample smoke and complete 54-window held-out evaluation. A2 then received a five-step QLoRA smoke, a 94-step/two-epoch SFT, adapter reload, and full evaluation. The same six test episodes were used for failure/recovery timing and the A2-versus-A3 Trace-Text ablation. GRPO completed five smoke updates and 100 full updates; GSPO completed the same budget as an **exploratory** comparison because GRPO had not improved the held-out primary task. Four RL checkpoints (25/50/75/100) and final adapters remain on the remote data disk. Local and remote suites each pass 40 tests.

The three Base variants all scored zero under the fixed strict JSON parser because their outputs were often code-fenced or used invalid phase labels. SFT improved State Macro-F1 to 0.4012 but still missed all 18 failure windows. GRPO and exploratory GSPO scored 0.1667 and 0.1619 State Macro-F1 respectively; both also had Failure Recall 0. Their higher recovery recall came with timing false alarms. See [the result table](artifacts/eval/final_model_comparison.csv), [paired Base/SFT errors](artifacts/eval/base_vs_sft.json), and [per-run evidence](artifacts/evidence/sft-eval-A2-20260930T014920Z/run_receipt.json).

Not executed: vision-language LoRA target ablation, a trace-trained adapter, or a full-suite cross-task holdout. These are not implied by the completed A2 pilot.

## V1 pilot result vs cross-task result

### Pilot result

- Data/schema audit, frozen split, A0–A3 materialization, Base/SFT/RL evaluations, timing, and trace ablation: executed with receipts.
- Best State Macro-F1: **SFT A2, 0.4012**; strict JSON validity **1.0**; Failure Recall **0**. No model met the failure-recognition objective.
- K=2 failure detection: **0/6** episodes for Base, SFT, GRPO, and exploratory GSPO; failure delay is **N/A** because there were no stable detections.

### Cross-task result

- Hold-out connector/mechanism evaluation: **N/A — the full REBOOT suite is not present in the active environment.**

A future cross-task run should train on selected mechanisms and hold out a connector or mechanism family such as threaded fasteners or a low-clearance keyed connector. Until that receipt exists, the project makes no transfer claim.

## Quick CPU verification

Use a project-local environment:

```bash
python -m venv .venv
# Windows
.venv/Scripts/python -m unittest discover -s tests -v
.venv/Scripts/python -m compileall -q src scripts tests

.venv/Scripts/python scripts/audit_reboot_data.py \
  --info artifacts/data_audit/source/meta/info.json \
  --phase artifacts/data_audit/source/meta/phase.json \
  --repository artifacts/data_audit/source/api.json \
  --source-endpoint https://hf-mirror.com

.venv/Scripts/python scripts/verify_local_dataset.py \
  --dataset-root artifacts/data_audit/source
```

Build or re-check manifests:

```bash
.venv/Scripts/python scripts/build_reboot_manifest.py \
  --phase-json artifacts/data_audit/source/meta/phase.json \
  --frames 4 \
  --output artifacts/manifests/reboot_windows.jsonl \
  --audit artifacts/manifests/reboot_windows_audit.json

.venv/Scripts/python scripts/build_failure_timing_manifest.py \
  --phase-json artifacts/data_audit/source/meta/phase.json \
  --split-manifest artifacts/splits/split_manifest.json
```

## GPU run

Use [`docs/AUTODL_RUNBOOK.md`](docs/AUTODL_RUNBOOK.md). The short form for Base A2 is:

```bash
.venv/bin/python scripts/eval_base.py \
  --model-root /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --dataset-root /root/autodl-tmp/prepared/A2 \
  --dataset-revision 0633573d0438be1185bddebdf1a2c8f5505f7b2a \
  --split test --num-frames 4 \
  --cameras observation.images.cam_high observation.images.cam_low \
  --load-in-4bit \
  --output-dir artifacts/runs/base-A2
```

Every formal evaluation run writes `predictions.jsonl`, `metrics.json`, `config.json`, `environment.json`, `stdout.log`, `stderr.log`, `timing.json`, `memory.json`, and `run_receipt.json`. Training runs additionally retain trainer logs/state, adapters, wall-clock time, peak VRAM, trainable parameter count, and output hashes.

Evaluate an SFT adapter with the same command plus `--adapter artifacts/runs/sft-main`; the base model path remains explicit so an adapter directory is never mistaken for a full model snapshot.

## Repository map

```text
src/reboot_recovery/        audit, splits, windows, rewards, metrics, gates, receipts
scripts/                    audit, preparation, Base/SFT/RLVR, timing, packaging
configs/                    pilot, evidence gates, GRPO and GSPO contracts
artifacts/data_audit/       pinned public metadata and generated audit receipts
artifacts/splits/           deterministic episode manifests
artifacts/eval/             executed comparisons, timing, and error cases
artifacts/evidence/         compact per-run predictions, logs, configs, and receipts
artifacts/materialization/  A0–A3 and timing materialization receipts
docs/architecture/          source-evidenced runtime architecture
docs/AUTODL_RUNBOOK.md      offline and GPU execution commands
upgraded_implementation/    legacy MathVista/chart provenance
```

The architecture diagram is a pre-GPU orientation snapshot. Its hardware-status note is historical; use the current experiment report and run receipts for measured results.

## Known limitations

- An earlier AutoDL container lacked CUDA; its probe is retained as historical evidence. The later RTX 4090 D host executed the reported GPU runs.
- The sample is one 16 mm cylinder-install task. It cannot establish cross-mechanism generalization.
- The pinned sample exposes four RGB streams but no depth feature, even though the full REBOOT project describes RGB-D collection.
- Two episodes have zero-duration failure intervals (`induced == recovery_started`); they produce nominal/recovery anchors but no failure anchor.
- Inclusive annotation terminals imply 53,880 rows, while the frame tables contain 53,886; the six extra frame-898 rows are quarantined and remain unexplained upstream.
- `SFT-VisionLanguage` and a trace-trained adapter were not run; GRPO and exploratory GSPO were memory-validated on the RTX 4090 D but did not improve failure recognition.
- Trace-Text is a deliberately simple first ablation; no numeric projector result exists.

## Legacy provenance

The original MathVista/chart GSPO reproduction remains under `upgraded_implementation/` with its original scripts and tests. It is not the primary runtime, dataset, or evidence source for REBOOT, and no chart result is presented as a robotics result.

## License and attribution

The pinned Hugging Face dataset metadata declares `apache-2.0`; the REBOOT project page currently states CC-BY-4.0. This repository records that discrepancy rather than choosing a license on the dataset authors' behalf. Review the upstream terms before redistributing data. Source code in this repository retains its existing project license status.
