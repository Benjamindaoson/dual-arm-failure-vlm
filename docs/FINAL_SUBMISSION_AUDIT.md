# WEBP submission audit — 2026-10-04

Target: [Everything Beneath the Policy (WEBP)](https://beneath-the-policy.github.io/),
CoRL 2026. The call permits at most four main-text pages (references and
appendices excluded), the CoRL 2026 submission template, and double-blind
review. Its deadline is October 9, 2026, 11:59 p.m. Central Time. This
repository holds an anonymous, locally compiled paper; it is **not** an
OpenReview submission. The machine-readable [audit](../artifacts/final_audit.json)
keeps `pdf_ready` separate from `submission_ready`.

## Claim and provenance boundary

The paper is an evaluation-protocol diagnostic, not a new detector or policy.
Its fixed-checkpoint contrast compares two receipt-verified Base validation
runs: the same model, C2 observations, four timestamps, decoding settings,
43 ordered IDs, references, and frozen split receipt. Recorded run configs
differ only in `task_schema`. The state-only output recognizes 15/15 failure
windows, whereas full phase/state/failure-mode output recognizes 0/15 after
the same narrow outer-fence normalization. The full-schema zero is semantic,
not a failure to parse normalized JSON. This establishes sensitivity to the
output formulation; it does **not** identify which changed field, instruction
length, or internal mechanism caused the difference.

The Tier-B comparison is distinct from that validation control. It uses six
frozen, internally exposed episodes, 54 windows, and identical sample IDs
across the saved Base/SFT predictions. State-only Base has strict JSON validity
0/54, failure recall 18/18, and nominal false alarms 18/18. Three SFT seeds
have strict validity 54/54 and failure recall 1/18, 7/18, and 7/18. The
individual seeds occupy different state-Macro-F1/false-alarm operating points;
none warrants an unconditional “SFT improves” or “SFT harms” conclusion.
Table 1, Table 2, and Figure 1 are generated from saved prediction files,
not hand-entered results. Paired intervals use 2,000 deterministic draws of
six whole episodes with identical draws for both conditions.

The episode exposure ledger covers the local 60-episode same-task sample:
seven are quarantined, 53 usable, and none untouched after earlier method
development. The final six Tier-B episodes are disjoint from the final refit's
fit/validation split but were exposed earlier. Thus the results are **internal
single-task evidence**, not independent or cross-task replication. A reward
intervention did not pass the V2 validation gate; no V2 GRPO or GSPO was run.
No production-safety or recovery-policy improvement is claimed.

## Verification ledger

| Check | Source and required outcome |
| --- | --- |
| Frozen experiment provenance | 44 completed V2 receipts verify; four final adapter hashes match training receipts. |
| Schema control | `scripts/audit_submission.py` independently checks both run receipts, ordered IDs/references/split hash, config-only schema difference, and recomputed normalized semantic metrics. |
| Tier-B primary evidence | `scripts/generate_final_paper_assets.py` verifies final runs and regenerates `paper/tables/final.tex`, `paper/tables/paired_contrasts.tex`, and `paper/figures/final_tradeoff.tex`. |
| Split exposure | `scripts/build_episode_exposure_ledger.py --validate-only` rebuilds the ledger against its receipt. |
| RL gate | `scripts/audit_submission.py` checks the validation prediction hashes and rejects any V2 RL run stage. |
| PDF source binding | `scripts/build_paper.py` requires committed Git blobs for all paper inputs, records the source commit, SHA-256 values, fixed build epoch, bibliography-start page, and PDF SHA-256. |
| Submission format | Default `corl_2026` mode, empty source author block, `Anonymous Submission` PDF metadata, and at most four main-text pages. |

The authoritative data records are
[final_paper_evidence.json](../artifacts/v2/final_paper_evidence.json),
the [episode-exposure receipt](../artifacts/episode_exposure_ledger_receipt.json),
the [verified-run summary](../artifacts/v2/verified_run_receipts.json), and
the local [PDF build receipt](../outputs/v2/paper_build/build_receipt.json).
The local submission [PDF](../outputs/v2/paper_build/main.pdf) is ignored by
Git; its exact source commit and hash are in that build receipt.

## Format, review, and remaining gate

The user-supplied official template is under
`corl_2026_template_submission/corl_2026_template_submission/`. Its default
style deliberately prints anonymous author placeholders, line numbers, and
the main-conference submission footer. For WEBP's double-blind review, these
are expected template artifacts, not missing real author metadata in the PDF.
Neither the style file nor the frozen evaluation/split semantics were edited
to change them. The `[preprint]` and `[final]` style modes are not used.

The public [WEBP OpenReview group](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Workshop/WEBP)
requires author profiles and form confirmations concerning email sharing and
release of accepted submissions. The user has said author information will
be provided later. No names, affiliations, profile IDs, or consent choices
were invented, and no PDF was uploaded. The anonymous PDF can be ready while
`submission_ready` remains false. Any eventual author-confirmed form record
belongs in ignored `outputs/v2/openreview_confirmation.json`, not in the
anonymous manuscript or Git.

## Reproduction and visual QA

From the repository root in its project-local `.venv`:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -q
    .\.venv\Scripts\python.exe -m compileall -q src scripts tests
    openspec validate webp-evaluation-substrate --strict
    .\.venv\Scripts\python.exe scripts\build_episode_exposure_ledger.py --validate-only
    .\.venv\Scripts\python.exe scripts\verify_v2_evidence.py --runs-root artifacts\v2\runs --output artifacts\v2\verified_run_receipts.json
    .\.venv\Scripts\python.exe scripts\generate_final_paper_assets.py --bootstrap-samples 2000
    .\.venv\Scripts\python.exe scripts\build_paper.py --tectonic <path-to-tectonic.exe>
    .\.venv\Scripts\python.exe scripts\audit_submission.py

Verification on October 4 passed: 90 regression tests, bytecode compilation,
strict OpenSpec validation, the 60-episode ledger check, all 44 completed run
receipts, and full evidence-asset regeneration. Two consecutive certified
Tectonic builds produced identical PDF SHA-256
`966fbd8860de0a43e58452a376aa98aa725a0fd54b74d2d62e0fbe41c7f40efa`
from source commit `4fee1f9`. `pdfinfo` reports five total pages and PDF Author
`Anonymous Submission`; the auxiliary bibliography label puts references on
page 5, so the main text occupies four pages. All five rendered pages were
inspected: tables, figure, citations, and visible URLs are legible and not
clipped. The default anonymous title block/footer is present by design.
The paper ends on a relatively sparse page 4; no layout rule was relaxed to
fill it. The local audit reports `pdf_ready=true` and
`submission_ready=false` solely because the OpenReview form needs author
confirmation. This is neither an OpenReview upload nor proof of acceptance.
