## ADDED Requirements

### Requirement: Immutable official dataset receipt
The system SHALL audit REBOOT metadata and annotations from a recorded repository id and resolved revision, and SHALL record hashes for every source file used.

#### Scenario: Metadata audit succeeds
- **WHEN** `meta/info.json` and `meta/phase.json` are available from a local root or configured Hugging Face endpoint
- **THEN** the system emits schema, annotation, phase, failure-mode, and Markdown receipts containing the resolved revision and source hashes

### Requirement: Invalid annotation quarantine
The system SHALL identify missing fields, duplicate episodes, out-of-range phases, reversed failure/recovery timing, inconsistent boundaries, and duration mismatches without silently repairing them.

#### Scenario: Annotation is inconsistent
- **WHEN** an episode violates any declared annotation invariant
- **THEN** the episode is listed with explicit reasons and excluded from train/evaluation manifests

### Requirement: Episode-safe deterministic split
The system SHALL assign complete episodes to 80/10/10 train/validation/test splits with seed 42 by default and SHALL emit a hashed manifest and leakage receipt.

#### Scenario: Split is generated twice
- **WHEN** the same valid episodes, ratios, and seed are supplied twice
- **THEN** both generated manifests and their SHA-256 values are identical and no episode appears in more than one split

### Requirement: Dense causal temporal windows
The system SHALL create at least three causal history windows for each available nominal, failure, and recovery interval while keeping every sampled frame at or before its labeled anchor.

#### Scenario: Pilot manifest is generated
- **WHEN** valid pilot episodes are transformed into model inputs
- **THEN** the manifest contains hundreds of windows, all windows inherit the episode split, and no sampled frame is later than its anchor
