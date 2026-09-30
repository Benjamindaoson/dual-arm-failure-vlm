# V2 evaluation protocol (frozen before new GPU runs)

The six cylinder episodes `02,08,18,20,47,54` are the `V1_DIAGNOSTIC_SET`. Their V1 predictions informed the research question and therefore cannot support an untouched V2 final-test claim. The original strict V1 metrics and prediction files remain unchanged.

## Two separate readouts

- **Strict protocol**: parse the naked JSON response with the existing exact-key, exact-taxonomy evaluator. A Markdown fence, prose, extra key, missing key, invalid label, or malformed JSON is invalid. This measures interface compliance.
- **Semantic diagnostic**: strip only leading/trailing whitespace; if the entire remainder is one lowercase `json` or unlabelled Markdown code fence, remove exactly that outer fence. Then call the *same* strict schema/label evaluator. No JSON repair, field editing, synonym mapping, prose search, or model judge is allowed. This measures recoverable task capability, not deployment compliance.

The semantic rule is fixed by `src/reboot_recovery/semantic_parser.py` and tested by `tests/test_semantic_parser.py`. Because it was motivated by inspection of V1 outputs, V1 semantic scores are explicitly retrospective diagnostics, never replacements for the frozen V1 official scores. All V2 model comparisons report both readouts.

Predicted-state probabilities, collapse ratio (`max_c P(c)`) and entropy (`-sum_c P(c) log P(c)`, natural logarithm) use valid taxonomy predictions as the denominator; invalid predictions are counted and reported separately. State macro-F1, class recall, precision, F1, balanced accuracy, and episode-bootstrap intervals are computed from prediction rows. Invalid outputs count as wrong for task metrics. Bootstrap units are episodes, never individual windows.
