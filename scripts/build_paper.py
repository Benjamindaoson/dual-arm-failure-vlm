"""Build the workshop PDF and bind it to its exact local source inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER_INPUTS = (
    "paper/main.tex",
    "paper/references.bib",
    "paper/corl_2026.sty",
    "paper/corlabbrvnat.bst",
    "paper/tables/final.tex",
    "paper/tables/paired_contrasts.tex",
    "paper/figures/final_tradeoff.tex",
)


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _pdf_pages(path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        raise RuntimeError("pdfinfo is required to certify PDF page count")
    result = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True,
                            errors="replace", check=True)
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("pdfinfo did not report a page count")


def verify_build_receipt(root: Path, observed_pages: int | None) -> dict:
    """Reject stale PDFs even when the output file exists and has four pages."""
    output = root / "outputs/v2/paper_build"
    receipt_path = output / "build_receipt.json"
    pdf = output / "main.pdf"
    if not receipt_path.is_file() or not pdf.is_file():
        raise ValueError("paper build receipt or PDF missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "BUILT" or receipt.get("pages") != observed_pages or observed_pages is None:
        raise ValueError("paper build receipt page count or status disagrees")
    if set(receipt.get("source_sha256", {})) != set(PAPER_INPUTS):
        raise ValueError("paper build source inventory differs")
    for relative, expected in receipt["source_sha256"].items():
        source = root / relative
        if not source.is_file() or _text_sha256(source) != expected:
            raise ValueError(f"paper build source changed: {relative}")
    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if receipt.get("pdf_sha256") != pdf_hash:
        raise ValueError("paper build PDF hash differs")
    return receipt


def build_paper(root: Path, tectonic: str) -> dict:
    root = root.resolve()
    for relative in PAPER_INPUTS:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    output = root / "outputs/v2/paper_build"
    output.mkdir(parents=True, exist_ok=True)
    command = [tectonic, "-X", "compile", "--outdir", str(output), "--outfmt", "pdf",
               "--print", "--untrusted", "main.tex"]
    result = subprocess.run(command, cwd=root / "paper", capture_output=True, text=True,
                            errors="replace", check=False)
    (output / "stdout.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Tectonic failed ({result.returncode}); see {output / 'stdout.log'}")
    pdf = output / "main.pdf"
    pages = _pdf_pages(pdf)
    if not 2 <= pages <= 4:
        raise ValueError(f"workshop PDF must have 2-4 pages: {pages}")
    version = subprocess.run([tectonic, "--version"], capture_output=True, text=True,
                             errors="replace", check=True).stdout.strip()
    receipt = {
        "status": "BUILT", "compiler": version, "command": command,
        "pages": pages,
        "source_sha256": {name: _text_sha256(root / name) for name in PAPER_INPUTS},
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    (output / "build_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
    verify_build_receipt(root, pages)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = parser.parse_args()
    if not args.tectonic:
        parser.error("Tectonic not on PATH; supply --tectonic with its executable path")
    receipt = build_paper(args.root, args.tectonic)
    print(json.dumps({"pages": receipt["pages"], "pdf_sha256": receipt["pdf_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
