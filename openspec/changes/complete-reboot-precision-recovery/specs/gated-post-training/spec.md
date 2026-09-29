## ADDED Requirements

### Requirement: Evidence-controlled stage decisions
The system SHALL centralize thresholds and emit machine-readable decisions for Base sufficiency, SFT value, and RL eligibility.

#### Scenario: Base meets sufficiency thresholds
- **WHEN** failure recall is at least 0.90 and state macro-F1 is at least 0.85
- **THEN** the decision is `BASE_SUFFICIENT` and later training stages are not authorized

### Requirement: Verifiable outcome reward
The reward SHALL be a class-aware weighted sum of correct phase, state, and applicable failure mode after JSON and taxonomy validation; formatting SHALL NOT provide a positive reward.

#### Scenario: Valid but wrong JSON is produced
- **WHEN** the object parses but all task labels are wrong
- **THEN** total reward is zero even though JSON validity is reported separately

### Requirement: Distinct GRPO and GSPO semantics
The launch configuration SHALL distinguish token-level GRPO from sequence-level GSPO and SHALL reject configurations that label Dr.GRPO loss as paper GSPO.

#### Scenario: RL launch is dry-run
- **WHEN** a valid SFT checkpoint, data path, algorithm, and short completion limit are configured with `--dry-run`
- **THEN** the resolved algorithm semantics and evidence gate inputs are printed without loading CUDA libraries
