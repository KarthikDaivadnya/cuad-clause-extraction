#!/usr/bin/env python3
"""
scripts/download_cuad.py
-------------------------
Downloads CUAD v1 (the official Atticus Project dataset) and extracts it
into data/. This is a separate, one-time setup script rather than being
baked into the pipeline itself, so re-running main.py never re-downloads
a 380MB archive by accident.

Source: Zenodo (official, DOI 10.5281/zenodo.4595826), mirrored on
Hugging Face (theatticusproject/cuad-qa) and GitHub (TheAtticusProject/cuad).`

After running this script you'll have:
    data/full_contract_pdf/   <- the PDFs main.py reads
    data/full_contract_txt/   <- plain-text versions (unused by this project,
                                 but handy for spot-checking extraction quality)
    data/CUAD_v1.json         <- the original expert clause annotations
    data/master_clauses.csv   <- flat CSV of all 41 label categories

Usage:
    python scripts/download_cuad.py                # downloads full dataset
    python scripts/download_cuad.py --sample 50     # then copies 50 PDFs into
                                                       data/full_contract_pdf_sample/
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

CUAD_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent                  # (root of this repo) represents as parent of this script
DATA_DIR = PROJECT_ROOT / "data"


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def make_sample(n: int):
    """Copy the first N PDFs (alphabetically) into a smaller sample folder,
    which is exactly what the assignment asks for: 'use a smaller subset,
    50 contracts'."""
    full_pdf_dir = DATA_DIR / "CUAD_v1" / "full_contract_pdf"
    if not full_pdf_dir.exists():
        full_pdf_dir = DATA_DIR / "full_contract_pdf"
    pdfs = sorted(full_pdf_dir.rglob("*.pdf")) + sorted(full_pdf_dir.rglob("*.PDF"))
    if not pdfs:
        print(f"[!] No PDFs found under {full_pdf_dir}", file=sys.stderr)
        sys.exit(1)

    sample_dir = DATA_DIR / "full_contract_pdf_sample"
    sample_dir.mkdir(exist_ok=True)
    for pdf in pdfs[:n]:
        shutil.copy(pdf, sample_dir / pdf.name)
    print(f"Copied {min(n, len(pdfs))} contracts into {sample_dir}")
    print(f"Run the pipeline with: python main.py --data-dir {sample_dir.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Download & prepare the CUAD dataset")
    parser.add_argument("--sample", type=int, default=50, help="Number of contracts to copy into a sample folder")
    parser.add_argument("--skip-download", action="store_true",
                         help="Skip download/extract (use if you already have data/CUAD_v1.zip extracted)")
    args = parser.parse_args()

    zip_path = DATA_DIR / "CUAD_v1.zip"

    if not args.skip_download:
        try:
            download_file(CUAD_ZIP_URL, zip_path)
            extract_zip(zip_path, DATA_DIR)
        except requests.exceptions.RequestException as exc:
            print(
                f"\n[!] Automated download failed ({exc}).\n"
                f"    Please download CUAD_v1.zip manually from:\n"
                f"      https://zenodo.org/records/4595826\n"
                f"      (mirror: https://huggingface.co/datasets/theatticusproject/cuad-qa)\n"
                f"    and unzip it into: {DATA_DIR}\n",
                file=sys.stderr,
            )
            sys.exit(1)

    make_sample(args.sample)


if __name__ == "__main__":
    main()
