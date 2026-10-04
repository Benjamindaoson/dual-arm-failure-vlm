# WEBP workshop paper source

`main.tex` is the double-blind submission draft for the CoRL 2026 workshop
[Everything Beneath the Policy (WEBP)](https://beneath-the-policy.github.io/).
It uses the user-supplied official `corl_2026` template in its default
anonymous submission mode. WEBP permits at most four **main-text** pages;
references and appendices are excluded. The current source deliberately uses
an empty `\author{}` block, and the resulting anonymous template title block,
line numbers, and main-conference submission footer are expected. Do not
switch to `[preprint]` or `[final]` for double-blind review, and do not edit the
official style to hide these elements.

From the repository root, use the project-local `.venv`:

    .\.venv\Scripts\python.exe scripts\generate_final_paper_assets.py --bootstrap-samples 2000
    .\.venv\Scripts\python.exe scripts\build_paper.py --tectonic <path-to-tectonic.exe>
    .\.venv\Scripts\python.exe scripts\audit_submission.py

The build requires each paper input to match a committed Git blob. It records
the source commit, fixed build epoch, PDF hash, and the bibliography-start
label used to verify the main-text page bound. The local PDF and build receipt
are under ignored `outputs/v2/paper_build/`; the machine audit is
`artifacts/final_audit.json`.

The paper is an internal single-task diagnostic. Its final Tier-B episodes
were exposed during earlier method development, and no V2 GRPO/GSPO or
independent cross-task replication is claimed. The fixed-checkpoint validation
schema contrast and final test metrics are separately identified in the text.

PDF readiness and OpenReview form readiness are distinct. The former can pass
without disclosing author identities. The latter requires the authors to
confirm their ordered names and OpenReview profile IDs plus the form's
email-sharing and data-release choices. If needed, place this private record
at ignored `outputs/v2/openreview_confirmation.json`; never commit it or put
identities in the double-blind source/PDF. No submission is made automatically.
