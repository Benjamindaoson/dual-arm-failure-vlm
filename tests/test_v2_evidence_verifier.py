import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.verify_v2_evidence import _hash_status


class EvidenceHashTests(unittest.TestCase):
    def test_crlf_checkout_is_recognized_without_relaxing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"state":"failure"}\r\n')
            expected = hashlib.sha256(b'{"state":"failure"}\n').hexdigest()
            self.assertEqual("CRLF_CHECKOUT_MATCH", _hash_status(path, expected))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                _hash_status(path, hashlib.sha256(b"wrong").hexdigest())


if __name__ == "__main__":
    unittest.main()
