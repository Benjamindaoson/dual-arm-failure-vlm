## Why

V1 strict scoring conflated interface compliance with task semantics: all 54 Base A2 outputs are fenced JSON, while mechanical fence removal exposes 6/18 correct failure states. The V1 SFT never predicts failure; GRPO predicts recovery for all 54 held-out windows. These are observed outcomes, not proof of a unique causal mechanism. V2 tests whether simpler supervision, wrist views, and train/test-matched trace input preserve failure evidence before any more RL.

## What Changes

- Freeze strict protocol and mechanically normalized semantic metrics as separate, never interchangeable readouts.
- Treat the six V1 test episodes as `V1_DIAGNOSTIC_SET`; run V2 screening on train/dev, then freeze a genuinely new final evaluation after methods are fixed.
- Run state-only Base/SFT, camera-count-matched wrist views, dense causal supervision, and trained trace comparison.
- Train both A2 and A3 full-schema SFT under the same V2 recipe for a fair trace comparison.
- Gate matched additive/state-gated GRPO on non-regressing validation failure recall, strict compliance, remaining errors, and verifier checks; preserve V1 RL evidence unchanged.
- Correct the V1 report's stale Git status and record V2 outcomes only when prediction receipts exist.

## Scope

The fixed sample is 60 REBOOT episodes with seven quarantines. V1 artifacts and split remain immutable. V2 may screen on reused episodes only as development evidence. A fresh episode-disjoint or officially sourced cross-task final set is required before confirmatory claims. New artifacts live only under `artifacts/v2/`, `outputs/v2/`, `docs/v2/`, and `paper/`.
