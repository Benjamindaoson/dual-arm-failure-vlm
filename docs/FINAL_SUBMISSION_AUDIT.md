# Workshop submission audit — 2026-10-04

Evidence package: **PASS WITH LIMITATIONS**. OpenReview submission readiness: **NO**, pending real author metadata and an author-visible build for the workshop's single-blind review. This is an auditable four-page anonymous checkpoint, not an uploaded final submission.

## What was verified

- The actual repository is Benjamindaoson/dual-arm-failure-vlm on master. Frozen final split and strict/semantic evaluators were not changed.
- The episode ledger has 60 rows: 53 usable and seven quarantined. All 53 usable cylinder-install episodes appear in V1 train, validation, or diagnostic test. The six final Tier-B episodes (11, 23, 29, 33, 51, 58) were in earlier method-development training. Thus the final result is **internal single-task evidence**, not independent replication.
- All 44 completed V2 run receipts verify. The four final adapter weight files match their training-receipt SHA-256 values. Six final prediction files reproduce their stored strict and semantic primary metrics; each sample ID and reference equals the pre-inference frozen test manifest and split receipt.
- The paper's final table, three-axis figure, and paired contrast table are generated from those saved predictions. The paired confidence intervals use 2,000 deterministic bootstrap draws over six *whole episodes* with identical paired draws, not independently sampled windows.
- The CoRL 2026 template was obtained from the official [CoRL author page](https://www.corl.org/contributions/instruction-for-authors), and the four-page PDF compiled. A local build receipt binds the PDF SHA-256 to the exact manuscript, bibliography, style, table, and figure inputs and verifies that each input equals the recorded Git commit's blob. The PDF build fixes `SOURCE_DATE_EPOCH` to the latest committed paper-source timestamp; two consecutive builds from the same source produced the same PDF SHA-256. All five cited works were checked against their arXiv or PMLR records. The [UMI Arena call](https://umi-arena.airoa.io/) confirms a 2–4 page extended abstract (references excluded), CoRL format, single-blind review, and October 16, 2026 paper deadline.
- The V2 RL gate artifact, its two validation prediction hashes, and all V2 run-stage configs were checked. The gate remains closed at REVISIT_REPRESENTATION_OR_SUPERVISION. No V2 GRPO or GSPO was run. The manuscript treats the state-gated reward as untested.

The authoritative machine report is [final_audit.json](../artifacts/final_audit.json); source hashes and generated-asset hashes are in [final_paper_evidence.json](../artifacts/v2/final_paper_evidence.json). The exposure ledger has its own [receipt](../artifacts/episode_exposure_ledger_receipt.json).

## Manuscript-result PASS/FAIL ledger

PASS means the result was recomputed from saved predictions or frozen source
metadata, not that it generalizes beyond this internal test. The final
evidence generator verifies each run receipt and prediction hash, binds all
54 IDs and references to the frozen test manifest/split receipt, compares
strict and narrowly normalized semantic metrics against saved metrics, and
compares every regenerated field with the saved evidence JSON. Its paired
intervals resample complete episodes with 2,000 draws and seed 42.

| Manuscript result | Check | Source |
| --- | --- | --- |
| Table 1: State Base | PASS | final-base-state prediction/metric/receipt/config files; final_paper_evidence.json |
| Table 1: State SFT seeds 42, 43, 44 | PASS, each seed | Three final-sft-state-seed*-test run directories; final_paper_evidence.json |
| Table 1: Full Base and Full SFT | PASS, each condition | final-base-full-test and final-sft-full-seed42-test run directories; final_paper_evidence.json |
| Table 1: strict JSON, semantic failure support/recall/precision, nominal false alarms, State Macro-F1, predicted-class distribution | PASS, every cell | scripts/generate_final_paper_assets.py regenerates paper/tables/final.tex from the six prediction files |
| Table 2: SFT 42, 43, 44 minus Base | PASS, each contrast and interval | scripts/paired_compare.py on paired final IDs; paper/tables/paired_contrasts.tex |
| Figure 1: strict JSON validity | PASS, all six bars | Strict parser output in final_paper_evidence.json; generated figure |
| Figure 1: semantic failure recall | PASS, all six bars | One-outer-fence semantic parser output in final_paper_evidence.json; generated figure |
| Figure 1: nominal-window false-alarm rate | PASS, all six bars | Nominal gold windows and semantic state predictions; generated figure |
| Text: 60 audited, seven quarantined, 53 usable, six internal Tier-B episodes | PASS | Frozen source audit, split receipts, episode_exposure_ledger.csv and its validated receipt |
| Text: two cameras, four past-only frames within the stated history | PASS | Final run configs and sampled-frame indices in the frozen final manifest |
| Text: 54 final windows with 18 per state; Base predicts failure in 53/54 and falsely alarms on 18/18 nominal windows | PASS | Frozen final manifest and State Base prediction rows |
| Text: five development-validation episodes and closed V2 RL gate | PASS | Frozen final split receipt and final_rl_gate_decision.json, bound to its validation prediction hashes |
| Independent or cross-task replication claim | NOT CLAIMED | All six Tier-B episodes had prior method-development exposure; no new task inference was run |

The three drawn measures are distinct; Figure 1 now also uses white,
dark-gray, and hatched fills so the legend remains interpretable without
color. No manuscript metric was hand-entered into a generated table or figure.

## Independent-evidence route decision

Route 1 is exhausted *within the locally audited 60-episode same-task sample*: zero usable episodes are untouched. This does **not** prove that no other cylinder-install episodes exist elsewhere.

Route 2 was not executed. Public REBOOT describes additional tasks, but locally held USB-C and RJ45 candidate folders contain only partial metadata; neither has a phase/onset annotation file or video payload. The USB-C episodes parquet copy lacks a valid closing PAR1 marker. M12 task labels were not locally inspected. This local environment has no nvidia-smi executable, no PyTorch in the project's .venv, and no local Qwen base weights. We did not freeze a new test set, tune on one, claim cross-task performance, or restart a paid GPU instance.

Route 3 therefore applies: retain an **internal diagnostic workshop study** and state that independent/cross-task replication is unmeasured.

## Format and visual QA

The [workshop call](https://umi-arena.airoa.io/) specifies a single-blind
2–4-page extended abstract, with acknowledgments/references excluded, in the
CoRL main-conference format. The [workshop OpenReview invitation](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Workshop/UMI_Arena)
was also inspected through its public API on October 4: it asks for author
profiles, title, abstract, PDF, and explicit confirmations concerning author
email sharing with program chairs and release of accepted submissions. The
actual authors and those confirmations are not available here; no form was
submitted.

The unmodified official style's default mode prints anonymous author fields
and a main-conference submission footer. Its preprint option displays the
supplied author list and omits that main-conference notice; its final option
prints a main-conference proceedings footer. For this single-blind workshop,
preprint is the defensible author-visible source option **by inference from
the supplied style and workshop call**, not an explicit workshop instruction
about the footer. The source must also override the style's default
anonymous PDF Author metadata with the real authors. Do not edit the style
file. Until real metadata is supplied, the current PDF remains a draft and
fails the author-visibility check. The audit also requires a separate
`paper/author_confirmation.json` containing an author-confirmed ordered list
of names, affiliations, and emails; each identity and its order must match
the rendered-source author block and the PDF Author string. The same record
must attest that the OpenReview profiles and the form's email-sharing and
accepted-paper public-release confirmations have been resolved. No such
confirmation exists yet, so this gate fails closed.

All four pages were rendered and inspected at normal reading size. PASS:
no overflowing text, clipped table/figure/caption, broken citation, broken
visible URL, or draft TODO was seen; Table 1, Table 2, and Figure 1 are
legible, and Figure 1's three encodings remain distinguishable in grayscale.
The bundled Tectonic build emitted nonfatal font/xcolor and intermediate-pass
citation warnings; the final PDF visibly resolves all five references.
FAIL for submission readiness: page 1 still has the template's anonymous
author block and main-conference submission footer, and PDF Author metadata
says Anonymous Submission. These are explicit unresolved placeholders,
not a silent QA pass.

## Claim hierarchy

- H1, interface compliance and semantic failure recognition need not track: **supported on internal evidence**.
- H2, this specific SFT setup can suppress the failure readout: **supported only in a recipe-specific sense**. Base's perfect failure recall is accompanied by maximal nominal false alarms and is not a good detector. Output formulation is implicated by a same-image Base prompt contrast, but the causal subcomponent is not isolated.
- H3, a state-gated reward prevents shortcut learning: **untested empirical intervention**; no V2 RL result.

No production robot safety, recovery-controller improvement, independent replication, cross-task generalization, or universal SFT degradation is claimed.

## Reproduction commands executed

Run from the repository root with the project-local Python environment:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -q
    .\.venv\Scripts\python.exe -m compileall -q src scripts tests
    openspec validate close-workshop-submission --strict
    .\.venv\Scripts\python.exe scripts\build_episode_exposure_ledger.py --validate-only
    .\.venv\Scripts\python.exe scripts\verify_v2_evidence.py --runs-root artifacts\v2\runs --output artifacts\v2\verified_run_receipts.json
    .\.venv\Scripts\python.exe scripts\generate_final_paper_assets.py --bootstrap-samples 2000
    .\.venv\Scripts\python.exe scripts\build_paper.py --tectonic <path-to-tectonic.exe>
    .\.venv\Scripts\python.exe scripts\audit_submission.py

The 88 regression tests, bytecode compile, OpenSpec validation, ledger validation, evidence verification, and final audit pass. The [verified run-receipt summary](../artifacts/v2/verified_run_receipts.json) records all 44 completed runs. The PDF was compiled with the Codex-bundled Tectonic executable (not an uninstalled global TeX package), and pdfinfo independently reports four pages. The ignored local [build receipt](../outputs/v2/paper_build/build_receipt.json) records compiler, source commit, source-input hashes, build epoch, and PDF hash. The PDF remains at [main.pdf](../outputs/v2/paper_build/main.pdf) in ignored local output storage. The exact source commit used for the final certified build is recorded in that receipt; uncommitted paper input cannot pass certification.

## Required before actual submission

The user will provide author names, affiliations, email addresses, and order later. Confirm that every author has the OpenReview profile required by the form, and obtain the form's email-sharing and public-release confirmations from the authors. Record an author-confirmed ordered identity list, replace the current anonymous draft metadata, switch the unmodified official style to its author-visible preprint mode, set matching non-anonymous PDF Author metadata, rebuild the PDF, inspect all pages, and rerun this audit. Until then, do **not** call the package submission-ready or upload the PDF.
