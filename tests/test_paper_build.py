import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import build_paper
from scripts.build_paper import PAPER_INPUTS, build_environment, verify_build_receipt
from scripts.generate_final_paper_assets import text_sha256


class PaperBuildTests(unittest.TestCase):
    def test_receipt_binds_pdf_to_every_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in PAPER_INPUTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "paper"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                            "commit", "-qm", "fixture"], cwd=root, check=True)
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip()
            pdf = root / "outputs/v2/paper_build/main.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-1.7\nfixture")
            (pdf.parent / "main.aux").write_text(
                r"\newlabel{referencesstart}{{5}{4}{Limitations}{section.5}{}}", encoding="utf-8")
            receipt = {
                "status": "BUILT", "pages": 4, "main_pages": 3,
                "git_commit": commit, "source_date_epoch": 1791111111,
                "source_sha256": {name: text_sha256(root / name) for name in PAPER_INPUTS},
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            }
            path = pdf.parent / "build_receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(verify_build_receipt(root, 4)["pdf_sha256"], receipt["pdf_sha256"])
            path.write_text(json.dumps({key: value for key, value in receipt.items() if key != "git_commit"}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "git commit"):
                verify_build_receipt(root, 4)
            path.write_text(json.dumps({**receipt, "git_commit": "b" * 40}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "git commit"):
                verify_build_receipt(root, 4)
            path.write_text(json.dumps(receipt), encoding="utf-8")
            (root / PAPER_INPUTS[2]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                verify_build_receipt(root, 4)

    def test_build_environment_fixes_pdf_timestamp(self):
        env = build_environment(1791111111)
        self.assertEqual(env["SOURCE_DATE_EPOCH"], "1791111111")

    def test_reference_start_label_counts_only_main_text_pages(self):
        parse = getattr(build_paper, "reference_start_page", None)
        self.assertIsNotNone(parse)
        self.assertEqual(parse(r"\newlabel{referencesstart}{{5}{5}{Limitations}{section.5}{}}"), 5)
        with self.assertRaisesRegex(ValueError, "referencesstart"):
            parse(r"\newlabel{another}{{}{5}}")


if __name__ == "__main__":
    unittest.main()
