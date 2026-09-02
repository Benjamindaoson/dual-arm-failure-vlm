## ADDED Requirements

### Requirement: Immutable multimodal course assets
The project SHALL preserve the notebook and baseline/after records with source hash metadata.

#### Scenario: Snapshot manifest
- **WHEN** material preparation is run
- **THEN** the manifest includes all three archived course assets and their SHA-256 values
