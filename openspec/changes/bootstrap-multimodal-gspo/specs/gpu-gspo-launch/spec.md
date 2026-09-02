## ADDED Requirements

### Requirement: Validated GSPO launch
The GSPO launcher SHALL validate model, dataset, output, and LoRA fields before importing optional vision-training packages.

#### Scenario: CPU dry run
- **WHEN** the launcher is invoked with `--dry-run`
- **THEN** it validates and prints the configuration without requiring a GPU
