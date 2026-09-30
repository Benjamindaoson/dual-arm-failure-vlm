## Context

The branch already has small annotation, reward, metric, Base-evaluation, and SFT prototypes, while legacy chart code remains in `upgraded_implementation/`. The public pilot is small (60 episodes), network access is unreliable, and the requested AutoDL target is an RTX 4090. The implementation needs an auditable standard-library evidence core plus real Base/SFT execution that fails closed if the allocated container does not expose CUDA.

## Goals / Non-Goals

**Goals:**

- Make official metadata and annotation provenance explicit and reproducible.
- Keep all splits episode-disjoint and all reported results tied to prediction rows and run receipts.
- Use one shared structured-output, metrics, timing, and reward core across Base/SFT/RL comparisons.
- Support native Hugging Face, `HF_ENDPOINT`, and fully local dataset/model roots.
- Execute real Base and SFT experiments when CUDA is visible and preserve every prediction and run receipt.

**Non-Goals:**

- End-to-end robot action control, object detection, segmentation, or classical CV benchmarking.
- Fabricated recovery-action labels or results.
- RL execution before the Base/SFT evidence gates permit it.
- Rewriting or deleting legacy MathVista/GSPO provenance.

## Decisions

1. **Standard-library evidence core.** JSON/JSONL/CSV parsing, hashing, split logic, bootstrap sampling, quarantine, and receipts use Python's standard library. Optional heavy dependencies are imported only inside GPU/data-materialization paths. This keeps audit, tests, and dry-runs usable offline.
2. **Immutable source receipts.** Audit inputs record repository id, resolved revision, source files, byte hashes, and acquisition endpoint. Generated statistics derive from downloaded metadata rather than README constants.
3. **Episode is the unit of independence.** Split assignment hashes a seeded, sorted episode list once. Windows inherit their episode split and validators reject cross-split episode reuse.
4. **JSON validity is a gate.** Invalid structured output earns zero RL reward and is separately reported by evaluation. Capability metrics operate on parsed task labels only.
5. **One run directory per formal execution.** Each formal run stores config, environment, predictions/logs, metrics, timestamps, and SHA-256 output hashes under `artifacts/runs/<run_id>/`.
6. **Distinct RL variants.** GRPO uses token-level importance sampling; GSPO uses sequence-level importance sampling. Both use the same SFT checkpoint, samples, short completions, and verifiable reward. Configuration validation prevents `dr_grpo` from being mislabeled as GSPO.
7. **Trace starts as text.** The first ablation summarizes recent numeric state/action changes compactly. A learned numeric projector remains out of scope until the text ablation shows held-out value.

## Risks / Trade-offs

- **Official endpoint unavailable** → support `HF_ENDPOINT`, immutable local snapshots, and receipts that state the endpoint used.
- **Pilot too small for stable conclusions** → report per-episode bootstrap intervals and label all sample-only evidence as pilot evidence.
- **Annotation inconsistencies** → quarantine instead of repairing; expose every reason in the audit.
- **TRL API drift** → pin/document tested version ranges and validate launch configs before importing GPU libraries.
- **Requested GPU is absent from the allocated container** → preserve the platform evidence, finish data/model preparation, and fail closed rather than emitting CPU-derived model results.

## Migration Plan

1. Add the new evidence core and tests beside existing modules.
2. Generate official pilot audit and split receipts from immutable metadata.
3. Route scripts through shared modules while keeping legacy chart paths untouched.
4. Download and verify the full pilot and pinned model under `/root/autodl-tmp`.
5. Run Base, gated SFT, timing, and trace evaluations on the CUDA host; run RL only if the gate permits it.
6. Validate evidence, commit, push, and open the unmerged PR.

Rollback is a normal Git revert of this change; no external schema or production service is mutated.

## Open Questions

- The full-suite cross-task holdout cannot be selected until the full REBOOT task inventory and local data are available; the repository will expose the contract and mark execution as pending.
- Vision-language LoRA and RL remain evidence-gated behind the first stable language-LoRA run.
