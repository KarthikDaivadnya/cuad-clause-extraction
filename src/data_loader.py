"""
data_loader.py
---------------
Task 1 of the assignment: "Load a subset of contracts from CUAD / Extract the
full contract text from PDF files."

Design notes:
    - CUAD_v1.zip ships a `full_contract_pdf/` folder (PDFs, organized by
      contract category) and a `full_contract_txt/` folder (pre-extracted
      text). We deliberately extract from the PDFs ourselves (as the
      assignment asks) rather than reading the provided .txt files, using
      pdfplumber for robust text extraction.
    - Contracts are walked recursively so the nested category folders in
      CUAD (Part_I/Part_II, then contract-type subfolders) don't matter.
    - `load_contracts` is generator-based and capped by `limit`, so scaling
      from 5 contracts (smoke test) to 50 (assignment spec) to all 510 is a
      one-line change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class Contract:
    contract_id: str
    source_path: str
    raw_text: str


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text page-by-page with pdfplumber, skipping unreadable pages
    instead of failing the whole document (some CUAD PDFs have scanned pages
    with no text layer)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract page %d of %s: %s", i, pdf_path.name, exc)
                text = ""
            pages.append(text)
    return "\n".join(pages)


def load_contracts(data_dir: Path, limit: Optional[int] = 50) -> Iterator[Contract]:
    """Yield Contract objects for up to `limit` PDFs found under data_dir.

    Args:
        data_dir: root folder containing CUAD PDFs (searched recursively).
        limit: max number of contracts to load. None = load all found.
    """
    data_dir = Path(data_dir)
    pdf_paths = sorted(data_dir.rglob("*.PDF")) + sorted(data_dir.rglob("*.pdf"))
    # de-dupe (case-insensitive filesystems can return both matches)
    pdf_paths = sorted(set(pdf_paths))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found under {data_dir}. "
            "Run `python scripts/download_cuad.py` first, or point --data-dir "
            "at your CUAD full_contract_pdf folder."
        )

    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    logger.info("Loading %d contract(s) from %s", len(pdf_paths), data_dir)

    for pdf_path in pdf_paths:
        contract_id = pdf_path.stem
        try:
            text = _extract_pdf_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping %s — could not read PDF (%s)", pdf_path.name, exc)
            continue
        if not text.strip():
            logger.warning("Skipping %s — no extractable text (likely scanned image)", pdf_path.name)
            continue
        yield Contract(contract_id=contract_id, source_path=str(pdf_path), raw_text=text)
