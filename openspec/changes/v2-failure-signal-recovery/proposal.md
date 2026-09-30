## Why

V1 strict scoring conflated interface compliance with task semantics: all 54 Base A2 outputs are fenced JSON, while mechanical fence removal exposes 6/18 correct failure states. The V1 SFT never predicts failure; GRPO predicts recovery for all 54 held-out windows. These are observed outcomes, not proof of a unique causal mechanism. V2 tests whether simpler supervision, wrist views, and train/test-matched trace input preserve failure evidence before any more RL.

## What Changes

- Freeze strict protocol and mechanically normalized semantic metrics as separate, never interchangeable readouts.
- Run state-only Base/SFT and a camera-count-matched wrist-view comparison on the existing episode-safe split.
- Train both A2 and A3 full-schema SFT under the same V2 recipe for a fair trace comparison.
- Gate any state-gated GRPO on validation failure recall and verifier checks; preserve V1 RL evidence unchanged.
- Correct the V1 report's stale Git status and record V2 outcomes only when prediction receipts exist.

## Scope

The fixed sample is 60 REBOOT episodes, with the original seven quarantines and 42/5/6 episode split. V2 does not re-audit, re-split, claim cross-task generalization, or tune V1 GRPO/GSPO. Reusing V1's six test episodes after inspecting their predictions makes V2 an exploratory pilot, not a fresh independent confirmation.
