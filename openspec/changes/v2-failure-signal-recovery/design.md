## Hypotheses and variants

1. **P0 — protocol versus semantics.** Re-score immutable V1 predictions under strict naked JSON and under exact outer lowercase `json` or unlabelled fence removal. The latter is retrospective for V1, frozen for future V2. It never changes keys or label values.
2. **P1 — state-only supervision.** Compare Base and NF4 QLoRA SFT on identical A2 high/low images, with only `{"state": ...}` as target. Keep V1 model revision, seed 42, two epochs, rank 16, learning rate, and image budget. Select checkpoints on validation loss, not the test set.
3. **P2 — local contact view.** Replace high/low with left/right wrist cameras while holding four timestamps, two cameras, sample IDs, state-only target, SFT recipe, and frozen split fixed. Train a separate adapter; inference-only camera swapping is not evidence for this question.
4. **P3 — trained trace.** Retrain both full-schema A2 and A3 adapters under one recipe and compare on their matching test inputs. The existing A2-trained/A3-tested result is only distribution-shift evidence.
5. **P4 — gated hierarchical RL.** State-only improvements authorize investigating full-schema supervision, not RL from a one-key checkpoint. Formal RL requires semantic failure recall above zero and no lower than matching Base, non-regressing macro-F1, strict JSON validity at least 0.95, remaining outcome errors and verified reward. From one SFT checkpoint compare matched additive and state-gated GRPO; missed failure gets a negative penalty. GSPO follows only a measured gated-GRPO held-out gain.

## Frozen measurement contract

- **Protocol metric:** existing strict parser, exact naked JSON schema and taxonomy.
- **Semantic metric:** trim outer whitespace; if and only if the entire remainder is one lowercase `json` or unlabelled Markdown fence, remove its opening and closing lines. Do not recover prose, repair JSON, remap labels, add fields, or change unknown taxonomy. Report normalization counts and strict validity separately.
- **Primary metric:** failure recall with numerator/support, not only a decimal. Secondary: state macro-F1, nominal/recovery recall, per-class prediction counts, full-schema phase/mode metrics where applicable, strict/semantic JSON validity, and exact three-field accuracy where applicable.
- **Unit of independence:** episode. Preserve prediction rows, run config, model/dataset revisions, split hash, seed, timing, VRAM, and output hashes. The fixed test set has six episodes and 18 failure windows; episode bootstrap intervals are descriptive. No significance or deployment claim is warranted by this size or by test reuse.

## Gate and stopping rules

- Compare state-only A2 and wrist candidates to matching Base controls on validation. Require failure recall > 0 and an improvement of at least one validation failure window to justify a later full-schema failure-supervision stage. Report test results for every preregistered variant even if this screen fails.
- The full-schema RL gate also requires non-regressing state macro-F1 and no increase in nominal-to-failure false-positive rate. If no checkpoint clears it, mark `REVISIT_REPRESENTATION_OR_SUPERVISION` and do not launch V2 RL. A3 full-schema SFT remains a separate trace question, not an RL workaround.
- If a candidate clears, run only a bounded five-step GRPO smoke first; full RL requires finite loss/gradients, adapter reload, a deterministic verifier, and no failure-class collapse on validation. Set a fixed update budget before launch and never promote training reward to held-out improvement.

## Sample size, risks, and execution

The V1 372/43/54 windows are diagnostic only for V2 method selection. The final test must be a new official task/episodes, or a newly frozen within-task split excluding the old six diagnostic episodes, with all final models retrained. Episode bootstrap intervals are descriptive for this small study. Risks include visual non-identifiability, class collapse, reward shortcuts, and cross-task availability; all negative results are retained.
