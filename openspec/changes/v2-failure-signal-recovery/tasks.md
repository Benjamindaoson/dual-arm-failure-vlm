## P0: Frozen diagnostic

- [x] Add test-first mechanical normalization and paired strict/semantic scorer.
- [x] Recompute V1 Base/SFT/GRPO/GSPO from committed raw predictions, preserving V1 formal scores.
- [x] Correct the V1 report's Git status and label the V2 diagnostic retrospective.

## P1–P3: Matched supervision and input experiments

- [x] Add test-first state-only prompt, target, parser, and metrics without changing full-schema defaults.
- [ ] Run A2 state-only Base and two-epoch QLoRA SFT on validation and frozen test. The executed final refit instead used validation-selected C2; three C2 SFT seeds and Base were evaluated on internal Tier B test.
- [ ] Materialize wrist-pair windows from existing local REBOOT data; train/evaluate a matched state-only adapter on frozen test. C0/C1/C2 were screened only on development validation, where C2 was selected.
- [ ] Retrain full-schema A2 and A3 adapters with identical recipes; compare paired test predictions. Executed controls include full-schema C2 final Base/SFT and trained Trace-Text on development validation; a full-schema final Trace-Text test does not exist.

## P4: Evidence-gated RL and verification

- [x] Evaluate the V2 gate on matching final validation predictions; decision `REVISIT_REPRESENTATION_OR_SUPERVISION` stops V2 RL.
- [x] Test and dry-run the state-gated reward and V2-only GRPO configuration.
- [x] Apply the conditional GRPO rule: the gate closed, so no V2 GRPO or GSPO was run. Historical V1 RL is diagnostic only.
- [x] Verify 15 final run receipts, all four final adapter hashes, paired predictions, wall-clock/VRAM, and the local test suite; report only executed results and the internal-Tier-B limitation.

The three unchecked P1–P3 items remain explicit gaps in the original proposal, not silently redefined as completed by the narrower final refit. They are not needed to interpret the reported C2 and full-schema evidence and were not run after the RL gate closed.
