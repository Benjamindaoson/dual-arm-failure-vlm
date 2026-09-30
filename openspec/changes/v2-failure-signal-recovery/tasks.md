## P0: Frozen diagnostic

- [x] Add test-first mechanical normalization and paired strict/semantic scorer.
- [x] Recompute V1 Base/SFT/GRPO/GSPO from committed raw predictions, preserving V1 formal scores.
- [x] Correct the V1 report's Git status and label the V2 diagnostic retrospective.

## P1–P3: Matched supervision and input experiments

- [x] Add test-first state-only prompt, target, parser, and metrics without changing full-schema defaults.
- [ ] Run A2 state-only Base and two-epoch QLoRA SFT on validation and frozen test.
- [ ] Materialize wrist-pair windows from existing local REBOOT data; train/evaluate a matched state-only adapter.
- [ ] Retrain full-schema A2 and A3 adapters with identical recipes; compare paired test predictions.

## P4: Evidence-gated RL and verification

- [ ] Evaluate the V2 gate on matching validation predictions; stop RL if failure signal is absent or worse than Base.
- [x] Test and dry-run the state-gated reward and V2-only GRPO configuration.
- [ ] If gate opens, run bounded GRPO smoke, then full GRPO only after finite/reload/verifier checks.
- [ ] Verify raw predictions, hashes, tests, wall-clock/VRAM, and update the report with only executed results.
