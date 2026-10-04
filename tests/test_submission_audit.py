import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from scripts.audit_submission import audit_submission, verify_recomputed_evidence, verify_rl_gate


class SubmissionAuditTests(unittest.TestCase):
    def test_current_saved_evidence_passes_but_submission_needs_authors(self):
        root = Path(__file__).resolve().parents[1]
        report = audit_submission(root)
        self.assertEqual(report["evidence_status"], "PASS_WITH_LIMITATIONS")
        self.assertFalse(report["submission_ready"])
        self.assertEqual(report["completed_v2_runs_verified"], 44)
        self.assertEqual(report["episode_exposure"]["untouched_same_task"], 0)
        self.assertEqual(report["adapter_hashes_verified"], 4)
        self.assertEqual(report["pdf_pages"], 4)
        self.assertIn("author metadata", report["submission_blockers"])

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


if __name__ == "__main__":
    unittest.main()
