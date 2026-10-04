# Workshop paper source

paper/main.tex is the four-page evidence-qualified draft for the CoRL 2026
UMI Arena workshop. It uses the official CoRL style. The current draft remains
anonymous because the actual author list, affiliations, and order have not yet
been supplied; it is **not** a single-blind submission PDF.

From the repository root, regenerate Table 1, Table 2, and Figure 1 from
receipt-verified final predictions with:

    .\.venv\Scripts\python.exe scripts\generate_final_paper_assets.py --bootstrap-samples 2000

Build the PDF and its source-hash receipt with:

    .\.venv\Scripts\python.exe scripts\build_paper.py --tectonic <path-to-tectonic.exe>

The build requires every paper input to match a committed Git blob. Its
receipt records that source commit and a fixed build epoch, so rebuilding the
same source yields identical PDF bytes.

Then run the final audit:

    .\.venv\Scripts\python.exe scripts\audit_submission.py

The final Tier-B episodes are disjoint from the final refit's fit split but
were exposed during earlier development; all reported results are internal
single-task diagnostics. No V2 controlled RL or cross-task result is claimed.
Only after actual author metadata is supplied should the official style's
author-visible preprint mode be used for the workshop submission; do not edit
the style file to remove its main-conference notice or anonymous title block.
Before claiming submission readiness, record the author-confirmed names,
affiliations, emails, and order in `paper/author_confirmation.json`, ensure
the manuscript author block follows that order, and set matching PDF Author
metadata. The audit remains blocked without this confirmation.
The confirmation record must also mark the workshop OpenReview profiles,
email-sharing confirmation, and accepted-paper public-release confirmation
as resolved before `submission_ready` can become true.
