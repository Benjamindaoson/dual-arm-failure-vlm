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

### Requirement: Final submission presentation remains evidence-qualified
The manuscript SHALL identify each reported result with its saved-artifact
provenance, distinguish past-only observation from causal identification, and
present interface validity, failure recall, and nominal false alarms as
separate measures. The PDF SHALL use the official CoRL style without editing
the style file to simulate author visibility. Its build receipt SHALL bind
the rendered inputs to the recorded Git commit and use a repeatable build
epoch. Submission readiness SHALL require an author-confirmed ordered identity
list matching the manuscript and PDF metadata.
The submission-ready gate SHALL also require confirmation of the workshop
OpenReview profile and author-consent prerequisites.

#### Scenario: Authors are not yet supplied
- **WHEN** real names, affiliations, and author order are unavailable
- **THEN** the audit marks submission readiness false and the draft is not
  uploaded or described as a single-blind final PDF.

#### Scenario: Paper input differs from the claimed build commit
- **WHEN** any manuscript, table, figure, bibliography, or style input differs
  from its blob at the receipt's Git commit
- **THEN** the build cannot be certified as an exact-commit reproducible PDF.
