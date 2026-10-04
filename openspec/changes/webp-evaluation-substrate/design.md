## Context

The repository has a four-page CoRL-style UMI Arena draft, frozen Tier-B predictions, a deterministic figure/table generator, and a source-bound PDF receipt. The user-provided CoRL 2026 submission archive matches the copied style semantically (the style-file differences are whitespace only). The official WEBP call requires a double-blind CoRL-submission or IEEE conference template and at most four pages of main text; references may extend beyond that. The current audit instead treats an author-visible preprint as submission-ready.

## Goals / Non-Goals

**Goals:** Rewrite the existing paper around evaluation substrate; independently verify the same-checkpoint validation schema contrast; build a double-blind CoRL PDF; report PDF readiness separately from OpenReview form readiness; preserve exact source and metric provenance.

**Non-Goals:** New inference or training, metric/split/evaluator changes, RL, independent replication, broad literature expansion, style-file or layout hacks, and any guarantee of acceptance.

## Decisions

1. **Keep `paper/main.tex` as the canonical manuscript.** Git preserves the earlier single-blind draft. A second manuscript would duplicate table and figure inputs and invite drift.
2. **Use the official default `\usepackage{corl_2026}`.** The supplied template example explicitly identifies it as initial anonymous submission. Its anonymous header, line numbers, and CoRL submission footer are native template output; they are not manually removed. `final` and `preprint` reveal authors and are inappropriate for double-blind review.
3. **Audit schema sensitivity from saved runs.** The state-only and full-schema Base validation runs must have matching model/input/sampling configurations except `task_schema`, and identical 43 ordered IDs and references. Their verified receipts, semantic failure counts, and source hashes anchor the manuscript claim.
4. **Keep distinct readiness states.** A PDF may be double-blind and venue-format-ready without named author profiles and OpenReview form confirmations. `pdf_ready` reflects only the audited manuscript/PDF; `submission_ready` remains false until those external prerequisites are confirmed. No OpenReview upload is performed.
5. **Do not invent a new performance scalar.** The paper compares strict validity, state Macro-F1, failure recall, nominal false alarms, and prediction distribution as separate axes. Different axes support different improvement verdicts; this is not a same-metric causal reversal or proof of an internal mechanism.

## Risks / Trade-offs

- **Fixed-checkpoint claim overstated** → verify every configuration field and ordered reference before using it; report only output-formulation sensitivity.
- **Page overflow** → compile under the unmodified template, count main-text pages independently of references, and visually inspect every page.
- **Accidental author disclosure** → leave the template in default anonymous mode and check visible title block and PDF metadata.
- **Prior UMI audit becomes misleading** → update it to a clearly named WEBP audit and retain old state in Git history.

## Migration Plan

Add WEBP-focused tests first, revise manuscript and audit, commit paper inputs, build the PDF from that exact source commit, regenerate the audit, and run the full suite plus visual inspection. The previous UMI draft is recoverable from Git history; no frozen experimental artifact is migrated.

## Open Questions

The names and OpenReview profiles of the actual authors and the form's required confirmations remain unavailable. They block upload, not preparation of a double-blind PDF.
