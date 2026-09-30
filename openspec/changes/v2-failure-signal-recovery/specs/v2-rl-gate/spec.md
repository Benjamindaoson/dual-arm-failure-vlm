## ADDED Requirements

### Requirement: Failure-first V2 RL gate
V2 RL SHALL remain closed unless a matching full-schema validation comparison shows nonzero SFT failure recall, at least one additional correctly identified failure window over Base, non-regressing state macro-F1 and nominal-to-failure false-positive rate, and remaining outcome errors. A state-only checkpoint SHALL NOT directly seed full-schema RL. V1 gate receipts SHALL not be rewritten.

#### Scenario: State macro-F1 improves but failure recall remains zero
- **WHEN** SFT gains state macro-F1 while predicting no failure window
- **THEN** the V2 decision is `REVISIT_REPRESENTATION_OR_SUPERVISION`, not `RUN_RLVR`

### Requirement: State-gated hierarchical reward
If V2 GRPO is authorized, phase and failure-mode credit SHALL be unavailable whenever the predicted execution state is wrong. Invalid JSON SHALL receive no positive task reward; reward configuration SHALL be recorded with the run.

#### Scenario: Recovery prediction repeats the correct failure mode on a failure row
- **WHEN** the reference state is failure and a valid prediction says recovery with the correct failure mode
- **THEN** its task reward is zero, regardless of phase or mode agreement
