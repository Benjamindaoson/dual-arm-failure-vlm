## ADDED Requirements

### Requirement: Frozen evaluation-substrate claims are reproducible
The WEBP manuscript SHALL derive every quantitative statement from saved, receipt-verified predictions or frozen manifests. It SHALL identify the fixed-checkpoint schema contrast as a validation diagnostic and Tier-B as internally exposed evidence.

#### Scenario: Fixed-checkpoint schema contrast
- **WHEN** the state-only and full-schema Base validation runs are audited
- **THEN** their model and input configs SHALL match except `task_schema`, all ordered IDs and references SHALL match, and the semantic failure counts SHALL be recomputed as 15/15 and 0/15 respectively.

#### Scenario: Tier-B metric-dependent conclusion
- **WHEN** the final Table 1, Table 2, and Figure 1 are regenerated
- **THEN** the manuscript SHALL separately report strict JSON validity, semantic state/failure measures, nominal false alarms, and prediction distribution without declaring either Base or SFT an overall detector winner.

### Requirement: WEBP paper frames the non-policy substrate
The manuscript SHALL present the critic as a narrow model organism for output-contract, parser, and post-training interface effects. It SHALL make four contributions explicit: evaluation decomposition, fixed-checkpoint interface sensitivity, post-training diagnostic, and a compact reporting checklist. It SHALL not claim representation collapse, general SFT degradation, cross-task replication, production safety, or executed V2 RL.

#### Scenario: Reviewer reads page one
- **WHEN** a reader finishes the first page
- **THEN** the research object SHALL be the evaluation substrate, rather than a new failure-detection benchmark or an unfinished RL method.

### Requirement: WEBP submission PDF uses the official double-blind mode
The paper SHALL use the unmodified CoRL 2026 submission style in its documented anonymous mode, SHALL keep main text within four pages, and SHALL permit references outside that limit. The audit SHALL distinguish `pdf_ready` from `submission_ready` and fail closed on accidental author disclosure.

#### Scenario: Author metadata is unavailable
- **WHEN** no verified OpenReview author profiles and form confirmations have been supplied
- **THEN** a correctly anonymized PDF MAY be `pdf_ready`, but `submission_ready` SHALL be false and no upload SHALL occur.

#### Scenario: Template-generated footer and header
- **WHEN** the default official CoRL submission mode produces its anonymous header and conference-submission footer
- **THEN** those elements SHALL be recorded as expected template output, not removed by style edits or layout overrides.

### Requirement: Source-bound PDF can be rebuilt
The PDF receipt SHALL bind every manuscript, bibliography, table, figure, and style input to the exact Git source commit and record the PDF hash and page count.

#### Scenario: Paper input is uncommitted or changed
- **WHEN** any paper input differs from its Git blob at the claimed build commit
- **THEN** the build or receipt verification SHALL fail rather than certify the PDF.
