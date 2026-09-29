## 1. Official Data Evidence

- [x] 1.1 Fetch and pin official pilot metadata through a configurable endpoint or local root
- [x] 1.2 Implement schema and annotation audit with explicit quarantine reasons
- [x] 1.3 Generate required data-audit artifacts from the pinned metadata
- [x] 1.4 Generate deterministic episode split manifest and receipt with distributions and hashes

## 2. Evaluation Core

- [x] 2.1 Harden strict structured-output and taxonomy validation
- [x] 2.2 Implement complete metrics, per-class scores, episode bootstrap intervals, and timing slices
- [x] 2.3 Implement failure detection delay and pre-failure false-alarm analysis
- [x] 2.4 Implement shared A0-A3 input-ablation contracts and comparison artifacts
- [x] 2.5 Update Base evaluator to emit predictions, metrics, config, environment, and run receipt per run

## 3. Gated Post-training

- [x] 3.1 Centralize and test Base/SFT/RL evidence thresholds and decisions
- [x] 3.2 Complete class-aware verifiable reward and edge-case tests
- [x] 3.3 Complete SFT language and vision-language dry-run configurations with checkpoint resume receipts
- [x] 3.4 Add distinct validated GRPO and sequence-level GSPO launch configurations

## 4. Offline AutoDL and Evidence

- [x] 4.1 Implement local dataset verification and endpoint/local-root resolution
- [x] 4.2 Implement deterministic AutoDL bundle packaging and offline bootstrap
- [x] 4.3 Add reusable run receipt and artifact hashing support
- [x] 4.4 Write the AutoDL runbook with exact CPU and gated GPU commands

## 5. Documentation and Architecture

- [x] 5.1 Create and validate the repository architecture source and standalone HTML
- [x] 5.2 Rewrite README with executed, implemented-not-executed, limitations, and legacy sections
- [x] 5.3 Add Chinese resume bullets, interview stories, and deep-dive Q&A with claim boundaries

## 6. Verification and Delivery

- [x] 6.1 Add tests for parser, leakage, windows, quarantine, metrics, timing, local mode, config, resume, and dry-run
- [x] 6.2 Run all tests, compile/static checks, dry-runs, and remote CPU verification
- [x] 6.3 Review diff for secrets, dead code, duplicate implementation, and unsupported claims
- [x] 6.4 Commit, push `reboot-precision-recovery`, and create an unmerged pull request to the default branch

## 7. Real 4090 Experiment

- [x] 7.1 Download and verify the complete pinned pilot and Qwen model snapshot under `/root/autodl-tmp`
- [ ] 7.2 Materialize causal A0-A3 and failure/recovery-onset datasets from real frames and traces
- [ ] 7.3 Run full Base A0-A2 evaluation and apply the SFT gate
- [ ] 7.4 Run SFT smoke, full selected-input SFT, adapter reload, and paired held-out evaluation
- [ ] 7.5 Run timing and trace ablations, apply the RL gate, and execute GRPO/GSPO only if permitted
- [x] 7.6 Generate the Chinese experiment report and update interview materials from verified numbers

Items 7.2–7.5 remain unchecked because the supplied SSH container exposes no NVIDIA device (`BLOCKED_NO_CUDA`). A0 was fully materialized and A2 real-decode smoke passed; no GPU metric or training stage is represented as executed.
