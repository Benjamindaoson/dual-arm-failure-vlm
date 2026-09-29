## ADDED Requirements

### Requirement: Strict execution-critic output
The system SHALL parse only a JSON object with `phase`, `state`, and `failure_mode`, using annotation-derived phase and failure taxonomies and `none` for nominal output.

#### Scenario: Model emits malformed output
- **WHEN** output is not the required JSON object or contains an unknown label
- **THEN** it is invalid, receives no task reward, and remains present in prediction evidence

### Requirement: Unified held-out metrics
The evaluator SHALL report state macro-F1, failure and recovery recall, pre-failure false-positive rate, phase macro-F1, failure-mode macro-F1, JSON valid rate, per-class metrics, per-episode bootstrap confidence intervals, and timing slices.

#### Scenario: Predictions are evaluated
- **WHEN** prediction JSONL includes references, episode ids, frame positions, and raw model outputs
- **THEN** aggregate metrics are emitted without discarding invalid or incorrect rows

### Requirement: Input and timing ablations
The system SHALL support A0 single-frame, A1 one-camera temporal, A2 two-camera temporal, and A3 temporal-plus-trace variants on identical test sample IDs, and SHALL compute false alarms and sustained failure-detection delay at fixed offsets around failure and recovery onset.

#### Scenario: Ablation outputs are compared
- **WHEN** completed prediction files for multiple variants share the split receipt
- **THEN** JSON and CSV comparison artifacts are produced only after sample IDs and split-receipt hashes match
