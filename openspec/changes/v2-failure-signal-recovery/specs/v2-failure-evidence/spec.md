## ADDED Requirements

### Requirement: Two non-interchangeable output readouts
The evaluator SHALL preserve the V1 strict score and SHALL separately report semantic scores after only the predeclared outer whitespace and exact JSON-fence normalization. It SHALL never overwrite raw predictions or official V1 metrics.

#### Scenario: Entire output is fenced JSON
- **WHEN** a model emits a single `json` fenced block containing an otherwise valid task object
- **THEN** strict validity is false, semantic validity is true, and the answer fields remain byte-identical inside the fence

#### Scenario: Output contains prose or invalid labels
- **WHEN** text surrounds the fence, JSON is malformed, or a taxonomy label is invalid
- **THEN** semantic normalization does not repair it

### Requirement: State-only task is independently scored
State-only SFT SHALL train on exactly one `state` target and SHALL be evaluated with a one-key state parser, preserving every image, sample ID, reference, and split receipt.

#### Scenario: State-only output is valid
- **WHEN** the output is a naked JSON object with only `state` and a valid execution-state label
- **THEN** failure recall, per-class counts, and state macro-F1 are computed without fabricating phase or failure-mode predictions

### Requirement: Matched input ablations
The wrist comparison SHALL use two wrist cameras against two fixed cameras at the same four causal timestamps, with separate training per variant. The trace comparison SHALL train A2 and A3 adapters separately under the same full-schema recipe.

#### Scenario: Comparison inputs differ in sample provenance
- **WHEN** sample IDs, episode sets, or split-receipt hashes differ
- **THEN** the comparison fails closed rather than publishing paired metrics
