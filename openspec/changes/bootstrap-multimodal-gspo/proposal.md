## Why

The course Qwen3-VL GSPO notebook and its records are useful learning assets but are not a portable project with reproducible CPU evaluation or a validated GPU launch path. This change preserves the notebook while making its data contract and reward assumptions testable.

## What Changes

- Archive the course notebook and small baseline/after record files with source checksums.
- Add deterministic final-answer extraction, correctness/format reward, and record evaluation.
- Add a dry-runnable validated LoRA GSPO GPU launch path without downloading models locally.

## Capabilities

### New Capabilities
- `course-provenance`: Preserves notebook and evaluation-record origin metadata.
- `chart-answer-evaluation`: Validates chart-task records and scores answer correctness and format.
- `gpu-gspo-launch`: Validates GPU training configuration and optional dependencies.

### Modified Capabilities

- None.

## Impact

Adds a standalone Python project, tests, scripts, and documentation; it does not change the original notebook or download vision models.
