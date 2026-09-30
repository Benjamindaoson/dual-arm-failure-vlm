## Hypotheses and variants

1. **P0 — protocol versus semantics.** Re-score the immutable V1 predictions under strict naked JSON and under exact outer ` ```json ` fence removal. The latter is a retrospective diagnostic for V1, then a frozen rule for future V2 runs. It never changes keys or label values.
2. **P1 — state-only supervision.** Compare Base and NF4 QLoRA SFT on identical A2 high/low images, with only `{"state": ...}` as target. Keep V1 model revision, seed 42, two epochs, rank 16, learning rate, and image budget. Select checkpoints on validation loss, not the test set.
3. **P2 — local contact view.** Replace high/low with left/right wrist cameras while holding four timestamps, two cameras, sample IDs, state-only target, SFT recipe, and frozen split fixed. Train a separate adapter; inference-only camera swapping is not evidence for this question.
4. **P3 — trained trace.** Retrain both full-schema A2 and A3 adapters under one recipe and compare on their matching test inputs. The existing A2-trained/A3-tested result is only distribution-shift evidence.
5. **P4 — gated hierarchical RL.** State-only improvements authorize investigating full-schema supervision, not RL from a one-key checkpoint. Only a full-schema SFT checkpoint whose validation failure recall is nonzero and at least one correct failure window above a matching full-schema Base control, with verifier reliability and remaining errors, may seed state-gated GRPO. A wrong execution state receives no phase or failure-mode credit. No GSPO is planned.

## Frozen measurement contract

- **Protocol metric:** existing strict parser, exact naked JSON schema and taxonomy.
- **Semantic metric:** trim outer whitespace; if and only if the entire remainder is one lowercase `json` Markdown fence, remove its opening and closing lines. Do not recover prose, repair JSON, remap labels, add fields, or change unknown taxonomy. Report normalization counts and strict validity separately.
- **Primary metric:** failure recall with numerator/support, not only a decimal. Secondary: state macro-F1, nominal/recovery recall, per-class prediction counts, full-schema phase/mode metrics where applicable, strict/semantic JSON validity, and exact three-field accuracy where applicable.
- **Unit of independence:** episode. Preserve prediction rows, run config, model/dataset revisions, split hash, seed, timing, VRAM, and output hashes. The fixed test set has six episodes and 18 failure windows; episode bootstrap intervals are descriptive. No significance or deployment claim is warranted by this size or by test reuse.

## Gate and stopping rules

- Compare state-only A2 and wrist candidates to matching Base controls on validation. Require failure recall > 0 and an improvement of at least one validation failure window to justify a later full-schema failure-supervision stage. Report test results for every preregistered variant even if this screen fails.
- The full-schema RL gate also requires non-regressing state macro-F1 and no increase in nominal-to-failure false-positive rate. If no checkpoint clears it, mark `REVISIT_REPRESENTATION_OR_SUPERVISION` and do not launch V2 RL. A3 full-schema SFT remains a separate trace question, not an RL workaround.
- If a candidate clears, run only a bounded five-step GRPO smoke first; full RL requires finite loss/gradients, adapter reload, a deterministic verifier, and no failure-class collapse on validation. Set a fixed update budget before launch and never promote training reward to held-out improvement.

## Sample size, risks, and execution

All 372/43/54 train/validation/test windows are fixed by the V1 split; this is the entire available single-task pilot, not a power-calculated sample. The smallest observable change in test Failure Recall is 1/18. The GPU run duration is measured, not predicted from traffic. Risks: test reuse, annotation-defined failure not visible in images, class collapse, reward shortcut, and remote GPU unavailability. Mitigations: explicit exploratory label, camera-matched training, class-count reporting, state gating, and no fabricated result when the host is unavailable.
