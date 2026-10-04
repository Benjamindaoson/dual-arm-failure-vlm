## ADDED Requirements

### Requirement: Episode exposure is auditable
The system SHALL derive a per-episode ledger from frozen split and window
manifests, distinguish train/validation/diagnostic/final-test exposure, and
reject a claim of untouched replication for any episode with prior exposure.

#### Scenario: Final Tier-B overlap with development training
- **WHEN** a final-test episode appeared in a V1 training manifest
- **THEN** it is labeled internal evidence rather than independent replication.

### Requirement: Manuscript values are artifact-derived
The system SHALL generate reported final metrics and paired intervals from
saved prediction rows, resampling at the episode level, and record source
hashes. It SHALL not promote an unexecuted V2 RL intervention to a result.

#### Scenario: Reproducing final evidence
- **WHEN** the saved predictions and receipts are validated
- **THEN** tables and figures are regenerated deterministically without
  hand-entered final-test metrics.
