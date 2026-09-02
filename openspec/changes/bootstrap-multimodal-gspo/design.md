## Context

The course supplies a Qwen3-VL GSPO notebook and small result records, but model execution requires a GPU and the answer/format reward must be independently testable first.

## Goals / Non-Goals

**Goals:** preserve notebook provenance, validate chart records, deterministically extract final answers, score correctness and format, and validate the GPU launch interface.

**Non-Goals:** download Qwen3-VL locally, claim the archived results as new results, or run GSPO in the CPU phase.

## Decisions

- Keep notebook and records immutable under legacy reproduction.
- Use a small record contract and answer parser that understand `<answer>` and `Final Answer:` conventions.
- Score correctness and output format separately so reward failures can be diagnosed.
- Make GPU import deferred behind a dry-run configuration check.

## Risks / Trade-offs

- [Free-form answers are ambiguous] → normalize whitespace/case and preserve raw output for review.
- [Exact answer matching is task-dependent] → expose per-record correctness inputs rather than hiding a judge model.
- [GPU package versions drift] → fail before training if optional dependencies or CUDA are unavailable.

## Migration Plan

Create and test CPU assets first. On GPU, attach the model/dataset, run `--dry-run`, then run LoRA GSPO. The legacy notebook remains unchanged throughout.

## Open Questions

The production image and model checkpoint are provided only at GPU execution time.
