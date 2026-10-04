"""Build the workshop PDF and bind it to its exact local source inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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


def reference_start_page(aux_text: str) -> int:
    """Read the LaTeX label placed after the main-text float flush."""
    match = re.search(r"\\newlabel\{referencesstart\}\{\{[^{}]*\}\{([0-9]+)\}", aux_text)
    if not match:
        raise ValueError("referencesstart label missing from LaTeX auxiliary file")
    return int(match.group(1))


def build_environment(source_date_epoch: int) -> dict[str, str]:
    """Keep TeX PDF timestamps stable across repeated builds of the same sources."""
    if source_date_epoch <= 0:
        raise ValueError("source date epoch must be positive")
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    return env


def _git_value(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          errors="replace", check=True).stdout.strip()


def _committed_source_sha256(root: Path, commit: str, relative: str) -> str:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"paper build git commit lacks source: {relative}")
    return hashlib.sha256(result.stdout.replace(b"\r\n", b"\n")).hexdigest()


def verify_build_receipt(root: Path, observed_pages: int | None) -> dict:
    """Reject stale PDFs even when the output exists and meets the page bound."""
    output = root / "outputs/v2/paper_build"
    receipt_path = output / "build_receipt.json"
    pdf = output / "main.pdf"
    aux = output / "main.aux"
    if not receipt_path.is_file() or not pdf.is_file() or not aux.is_file():
        raise ValueError("paper build receipt, PDF, or auxiliary page label missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "BUILT" or receipt.get("pages") != observed_pages or observed_pages is None:
        raise ValueError("paper build receipt page count or status disagrees")
    first_reference_page = reference_start_page(aux.read_text(encoding="utf-8"))
    if (receipt.get("main_pages") != first_reference_page - 1
            or not 1 <= receipt["main_pages"] <= 4 or first_reference_page > observed_pages):
        raise ValueError("paper build main-text page count disagrees or exceeds WEBP limit")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("git_commit", ""))):
        raise ValueError("paper build git commit is missing or invalid")
    if not isinstance(receipt.get("source_date_epoch"), int) or receipt["source_date_epoch"] <= 0:
        raise ValueError("paper build source date epoch is missing or invalid")
    if set(receipt.get("source_sha256", {})) != set(PAPER_INPUTS):
        raise ValueError("paper build source inventory differs")
    for relative, expected in receipt["source_sha256"].items():
        source = root / relative
        if not source.is_file() or _text_sha256(source) != expected:
            raise ValueError(f"paper build source changed: {relative}")
        if _committed_source_sha256(root, receipt["git_commit"], relative) != expected:
            raise ValueError(f"paper build git commit differs from source: {relative}")
    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if receipt.get("pdf_sha256") != pdf_hash:
        raise ValueError("paper build PDF hash differs")
    return receipt


def build_paper(root: Path, tectonic: str) -> dict:
    root = root.resolve()
    git_commit = _git_value(root, "rev-parse", "HEAD")
    source_date_epoch = int(_git_value(root, "log", "-1", "--format=%ct", "--", "paper"))
    for relative in PAPER_INPUTS:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
        if _committed_source_sha256(root, git_commit, relative) != _text_sha256(root / relative):
            raise ValueError(f"paper build git commit differs from source: {relative}")
    output = root / "outputs/v2/paper_build"
    output.mkdir(parents=True, exist_ok=True)
    command = [tectonic, "-X", "compile", "--outdir", str(output), "--outfmt", "pdf",
               "--keep-intermediates",
               "--print", "--untrusted", "main.tex"]
    result = subprocess.run(command, cwd=root / "paper", capture_output=True, text=True,
                            errors="replace", check=False, env=build_environment(source_date_epoch))
    (output / "stdout.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Tectonic failed ({result.returncode}); see {output / 'stdout.log'}")
    pdf = output / "main.pdf"
    pages = _pdf_pages(pdf)
    first_reference_page = reference_start_page((output / "main.aux").read_text(encoding="utf-8"))
    main_pages = first_reference_page - 1
    if not 1 <= main_pages <= 4 or first_reference_page > pages:
        raise ValueError(f"WEBP main text must have at most 4 pages: {main_pages}")
    version = subprocess.run([tectonic, "--version"], capture_output=True, text=True,
                             errors="replace", check=True).stdout.strip()
    receipt = {
        "status": "BUILT", "compiler": version, "command": command,
        "pages": pages, "main_pages": main_pages,
        "git_commit": git_commit, "source_date_epoch": source_date_epoch,
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
    print(json.dumps({"pages": receipt["pages"], "main_pages": receipt["main_pages"],
                      "pdf_sha256": receipt["pdf_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
