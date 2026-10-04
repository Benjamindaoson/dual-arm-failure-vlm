import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_paper import PAPER_INPUTS, verify_build_receipt
from scripts.generate_final_paper_assets import text_sha256


class PaperBuildTests(unittest.TestCase):
    def test_receipt_binds_pdf_to_every_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in PAPER_INPUTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            pdf = root / "outputs/v2/paper_build/main.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-1.7\nfixture")
            receipt = {
                "status": "BUILT", "pages": 4,
                "source_sha256": {name: text_sha256(root / name) for name in PAPER_INPUTS},
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            }
            path = pdf.parent / "build_receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(verify_build_receipt(root, 4)["pdf_sha256"], receipt["pdf_sha256"])
            (root / PAPER_INPUTS[2]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                verify_build_receipt(root, 4)


if __name__ == "__main__":
    unittest.main()
