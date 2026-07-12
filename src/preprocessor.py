"""
preprocessor.py
----------------
Task 1 (continued): "Normalize text."

PDF-extracted legal text tends to have: hyphenated line-wraps, repeated
whitespace, page headers/footers bleeding into the body, and inconsistent
quote/dash characters. We clean these up before chunking, because messy
whitespace both wastes LLM tokens and hurts embedding-based retrieval.
"""
import re
import unicodedata


_PAGE_NUM_RE = re.compile(r"\n\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*\n", re.IGNORECASE)
_HYPHEN_WRAP_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_UNDERSCORE_RE = re.compile(r"_{3,}")  # signature-line underscores


def normalize_text(raw_text: str) -> str:
    """Clean raw PDF text into a normalized, LLM/embedding-friendly string."""
    text = unicodedata.normalize("NFKC", raw_text)

    # Standardize smart quotes / dashes so keyword search stays reliable.
    text = (
        text.replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
    )

    # Re-join words that were split across a line-wrap hyphen, e.g. "termina-\ntion" -> "termination"
    text = _HYPHEN_WRAP_RE.sub(r"\1\2", text)

    # Strip standalone page-number lines.
    text = _PAGE_NUM_RE.sub("\n", text)

    # Collapse long underscore runs used for signature blanks.
    text = _MULTI_UNDERSCORE_RE.sub("____", text)

    # Collapse whitespace.
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()
