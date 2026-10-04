# Workshop submission audit — 2026-10-04

Evidence package: **PASS WITH LIMITATIONS**. OpenReview submission readiness: **NO**, pending author metadata for the workshop's single-blind review. This is an auditable 4-page anonymous draft, not the uploaded final version.

## What was verified

- The actual repository is Benjamindaoson/dual-arm-failure-vlm on master. Frozen final split and strict/semantic evaluators were not changed.
- The episode ledger has 60 rows: 53 usable and seven quarantined. All 53 usable cylinder-install episodes appear in V1 train, validation, or diagnostic test. The six final Tier-B episodes (11, 23, 29, 33, 51, 58) were in earlier method-development training. Thus the final result is **internal single-task evidence**, not independent replication.
- All 44 completed V2 run receipts verify. The four final adapter weight files match their training-receipt SHA-256 values. Six final prediction files reproduce their stored strict and semantic primary metrics; each sample ID and reference equals the pre-inference frozen test manifest and split receipt.
- The paper's final table, three-axis figure, and paired contrast table are generated from those saved predictions. The paired confidence intervals use 2,000 deterministic bootstrap draws over six *whole episodes* with identical paired draws, not independently sampled windows.
- The CoRL 2026 template was obtained from the official [CoRL author page](https://www.corl.org/contributions/instruction-for-authors), and the 4-page PDF compiled. A local build receipt binds the PDF SHA-256 to the exact manuscript, bibliography, style, table, and figure inputs. The three cited paper records were checked on arXiv. The [UMI Arena call](https://umi-arena.airoa.io/) confirms a 2–4 page extended abstract, CoRL format, single-blind review, and October 16, 2026 paper deadline.
- The V2 RL gate artifact, its two validation prediction hashes, and all V2 run-stage configs were checked. The gate remains closed at REVISIT_REPRESENTATION_OR_SUPERVISION. No V2 GRPO or GSPO was run. The manuscript treats the state-gated reward as untested.

The authoritative machine report is [final_audit.json](../artifacts/final_audit.json); source hashes and generated-asset hashes are in [final_paper_evidence.json](../artifacts/v2/final_paper_evidence.json). The exposure ledger has its own [receipt](../artifacts/episode_exposure_ledger_receipt.json).

## Independent-evidence route decision

Route 1 is exhausted *within the locally audited 60-episode same-task sample*: zero usable episodes are untouched. This does **not** prove that no other cylinder-install episodes exist elsewhere.

Route 2 was not executed. Public REBOOT describes additional tasks, but locally held USB-C and RJ45 candidate folders contain only partial metadata; neither has a phase/onset annotation file or video payload. The USB-C episodes parquet copy lacks a valid closing PAR1 marker. M12 task labels were not locally inspected. This local environment has no nvidia-smi executable, no PyTorch in the project's .venv, and no local Qwen base weights. We did not freeze a new test set, tune on one, claim cross-task performance, or restart a paid GPU instance.

Route 3 therefore applies: retain an **internal diagnostic workshop study** and state that independent/cross-task replication is unmeasured.

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
    .\.venv\Scripts\python.exe scripts\verify_v2_evidence.py --runs-root artifacts\v2\runs --output outputs\v2\submission_baseline_receipts.json
    .\.venv\Scripts\python.exe scripts\generate_final_paper_assets.py --bootstrap-samples 2000
    .\.venv\Scripts\python.exe scripts\build_paper.py --tectonic <path-to-tectonic.exe>
    .\.venv\Scripts\python.exe scripts\audit_submission.py

The 83 regression tests, bytecode compile, OpenSpec validation, ledger validation, evidence verification, and final audit pass. The PDF was compiled with the Codex-bundled Tectonic executable (not an uninstalled global TeX package), and pdfinfo independently reports four pages. The ignored local [build receipt](../outputs/v2/paper_build/build_receipt.json) records compiler, input hashes, and PDF hash. The PDF remains at [main.pdf](../outputs/v2/paper_build/main.pdf) in ignored local output storage.

## Required before actual submission

The user will provide author names, affiliations, email addresses, and order later. Replace the current anonymous draft metadata, switch the official style to its author-visible final mode appropriate for this single-blind workshop, rebuild the PDF, and rerun this audit. Until then, do **not** call the package submission-ready or upload the PDF.
