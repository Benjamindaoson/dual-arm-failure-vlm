## ADDED Requirements

### Requirement: Offline data and model roots
All data, model, cache, and checkpoint scripts SHALL accept explicit local roots and SHALL honor `HF_ENDPOINT`; AutoDL defaults SHALL use `/root/autodl-tmp` for large assets.

#### Scenario: Network is unavailable
- **WHEN** required metadata, model, and dataset files already exist locally
- **THEN** audit, preparation, evaluation, and training commands can resolve them without contacting Hugging Face

### Requirement: Formal run receipt
Every formal run SHALL record run id, Git commit, config and output hashes, dataset/model revision, seed, environment versions, device data, timestamps, peak VRAM when applicable, and trainable parameter count when applicable.

#### Scenario: CPU dry-run completes
- **WHEN** a formal CPU-capable command completes
- **THEN** its run directory contains machine-readable config, environment, receipt, and output hashes while unavailable GPU fields are explicitly null

### Requirement: AutoDL bootstrap and verification
The repository SHALL provide a package command, local dataset verifier, and bootstrap script that do not store tokens in source control.

#### Scenario: Bundle is unpacked on AutoDL
- **WHEN** the bootstrap is run in an offline shell
- **THEN** cache variables target `/root/autodl-tmp`, a project-local environment is created, and CPU verification commands are available without writing large assets to the system disk
