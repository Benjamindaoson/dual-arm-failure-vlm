## Why

The target branch contains a promising REBOOT prototype but lacks the audited dataset receipts, complete evaluation surface, offline AutoDL workflow, reproducible run evidence, and tests required to support its claims. This change turns it into a CPU-verifiable and GPU-ready independent research repository while preserving the earlier MathVista/GSPO work as legacy provenance.

## What Changes

- Audit the official `REBOOT26/sample_recovery-demonstration` metadata and annotations at an immutable dataset revision, quarantine invalid episodes, and emit machine-readable receipts.
- Produce deterministic episode-level train/validation/test manifests with leakage checks, class distributions, and SHA-256 receipts.
- Standardize the failure-aware execution critic task, input ablations, prediction records, metrics, bootstrap confidence intervals, and failure-timing analysis.
- Add evidence gates for Base, SFT, and RLVR; implement verifiable rewards plus distinct GRPO and sequence-level GSPO launch configurations without treating format as capability.
- Add local/offline dataset verification, AutoDL packaging/bootstrap, run receipts, and resumable GPU commands that keep large assets under `/root/autodl-tmp`.
- Expand tests, rewrite the README around executed evidence, add architecture and interview artifacts, and retain legacy chart code outside the primary REBOOT path.

## Capabilities

### New Capabilities

- `reboot-data-evidence`: Immutable metadata/annotation audit, quarantine, episode-safe splits, and dataset receipts.
- `failure-aware-evaluation`: Structured critic samples, input ablations, metrics, bootstrap confidence intervals, and detection-delay evaluation.
- `gated-post-training`: Base/SFT/RL evidence gates, QLoRA preparation, verifiable rewards, and distinct GRPO/GSPO configurations.
- `offline-autodl-reproducibility`: Local dataset verification, packaging, environment bootstrap, run receipts, and offline execution documentation.

### Modified Capabilities

None.

## Impact

Primary changes affect `src/reboot_recovery/`, REBOOT scripts/configuration/tests, `artifacts/`, `docs/`, and `README.md`. Existing `upgraded_implementation/` and chart-oriented scripts/configuration remain as explicitly labeled legacy provenance.
