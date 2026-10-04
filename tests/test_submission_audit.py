import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from scripts.audit_submission import (
    audit_submission, verify_recomputed_evidence, verify_rl_gate,
)
from scripts import audit_submission as submission_module


class SubmissionAuditTests(unittest.TestCase):
    def test_current_saved_evidence_has_anonymous_webp_pdf_but_needs_form(self):
        root = Path(__file__).resolve().parents[1]
        report = audit_submission(root)
        self.assertEqual(report["evidence_status"], "PASS_WITH_LIMITATIONS")
        self.assertFalse(report["submission_ready"])
        self.assertEqual(report["completed_v2_runs_verified"], 44)
        self.assertEqual(report["episode_exposure"]["untouched_same_task"], 0)
        self.assertEqual(report["adapter_hashes_verified"], 4)
        self.assertTrue(report["pdf_ready"])
        self.assertEqual(report["pdf_pages"], 5)
        self.assertEqual(report["main_text_pages"], 4)
        self.assertEqual(report["fixed_checkpoint_schema_contrast"]["state_only_semantic_failure_correct"], 15)
        self.assertEqual(report["fixed_checkpoint_schema_contrast"]["full_schema_semantic_failure_correct"], 0)
        self.assertEqual(report["official_venue"]["review"], "double-blind")
        self.assertRegex(report["paper_build_git_commit"], r"^[0-9a-f]{40}$")
        self.assertIn("OpenReview form", report["submission_blockers"])

    def test_mismatched_generated_asset_is_rejected(self):
        from scripts.audit_submission import verify_hash_map
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.tex"
            path.write_text("wrong", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_hash_map(Path(directory), {"figure.tex": "0" * 64})

    def test_text_hash_accepts_checkout_newline_translation(self):
        import hashlib
        from scripts.audit_submission import verify_hash_map
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_bytes(b'{"a":1}\r\n')
            expected = hashlib.sha256(b'{"a":1}\n').hexdigest()
            verify_hash_map(Path(directory), {"metrics.json": expected})

    def test_full_recomputed_evidence_must_match_saved_json(self):
        evidence = {"runs": {"State Base": {"semantic": {"failure_correct": 18,
                     "state_macro_f1": 0.169}}}, "contrasts": {"State SFT 42": {"delta": -1}}}
        verify_recomputed_evidence(evidence, dict(evidence))
        altered = json.loads(json.dumps(evidence))
        altered["runs"]["State Base"]["semantic"]["state_macro_f1"] = 0.9
        with self.assertRaisesRegex(ValueError, "evidence"):
            verify_recomputed_evidence(altered, evidence)

    def test_rl_gate_must_be_closed_and_bound_to_validation_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "artifacts/v2/runs"
            paths = [run_root / name / "predictions.jsonl" for name in
                     ("final-base-full-val", "final-sft-full-seed42-val")]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")
            gate = {
                "status": "VALIDATION_GATE", "protocol": "V2",
                "decision": "REVISIT_REPRESENTATION_OR_SUPERVISION",
                "base_failure_correct": 0, "sft_failure_correct": 0,
                "failure_correct_gain": 0, "verifier_validated": True,
                "base_predictions_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                "sft_predictions_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
            }
            gate_path = root / "artifacts/v2/final_rl_gate_decision.json"
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            self.assertEqual(verify_rl_gate(root)["decision"], gate["decision"])
            gate["decision"] = "PROCEED_GRPO"
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "RL gate"):
                verify_rl_gate(root)

    def test_webp_pdf_requires_default_anonymous_style_and_no_source_identity(self):
        check = getattr(submission_module, "webp_pdf_blockers", None)
        self.assertIsNotNone(check)
        anonymous = r"\usepackage{corl_2026}" + "\n" + r"\author{}"
        self.assertEqual(check(anonymous, pdf_author="Anonymous Submission"), [])
        self.assertIn("double-blind style", check(anonymous.replace(
            r"\usepackage{corl_2026}", r"\usepackage[preprint]{corl_2026}"),
            pdf_author="Anonymous Submission"))
        self.assertIn("source author identity", check(anonymous.replace(
            r"\author{}", r"\author{Jane Doe}"), pdf_author="Anonymous Submission"))
        self.assertIn("PDF author metadata", check(anonymous, pdf_author="Jane Doe"))

    def test_openreview_upload_needs_private_author_profile_and_form_confirmations(self):
        check = getattr(submission_module, "openreview_form_blockers", None)
        self.assertIsNotNone(check)
        self.assertIn("OpenReview form", check(None))
        confirmation = {"confirmed_by_author": True, "authors": [
            {"name": "Jane Doe", "profile_id": "~Jane_Doe1", "order": 1}],
            "email_sharing_confirmed": True, "data_release_confirmed": True}
        self.assertEqual(check(confirmation), [])
        self.assertIn("OpenReview form", check({**confirmation, "data_release_confirmed": False}))


if __name__ == "__main__":
    unittest.main()
