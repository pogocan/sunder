"""
sunder.extract -- PDF text extraction using PyPDF directly.

No Haystack dependency. Returns text with page break markers.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .constants import PAGE_BREAK

# Dehyphenation: rejoin words split across line breaks (e.g. "knowl-\nedge" -> "knowledge")
_RE_DEHYPHEN = re.compile(r'(\w)-\s*\n\s*(\w)')

# Collapse runs of 3+ newlines down to 2
_RE_EXCESS_NEWLINES = re.compile(r'\n{3,}')

# Collapse runs of 2+ spaces (not newlines) down to 1
_RE_EXCESS_SPACES = re.compile(r'[^\S\n]{2,}')


def clean_text(text: str) -> str:
    """Clean extracted PDF text.

    - Rejoin hyphenated line breaks (e.g. "knowl-\\nedge" -> "knowledge")
    - Collapse excessive whitespace (multiple spaces -> one, 3+ newlines -> 2)
    """
    text = _RE_DEHYPHEN.sub(r'\1\2', text)
    text = _RE_EXCESS_SPACES.sub(' ', text)
    text = _RE_EXCESS_NEWLINES.sub('\n\n', text)
    return text.strip()


def extract_pdf(
    pdf_path: str | Path,
    pages: str | tuple[int, int] | None = None,
    clean: bool = True,
) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        pages: Optional page range. Accepts:
               - String: "1-50" or "10" (1-indexed, inclusive)
               - Tuple: (1, 50) (1-indexed, inclusive)
               - None: all pages
        clean: Apply text cleaning (dehyphenation, whitespace normalization). Default True.

    Returns:
        Full text with page break markers between pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # Determine page range
    if isinstance(pages, tuple):
        start = max(1, pages[0])
        end = min(pages[1], total_pages)
    elif pages:
        start, end = _parse_pages_range(pages, total_pages)
    else:
        start, end = 1, total_pages

    # Extract text from each page
    page_texts = []
    for i in range(start - 1, end):
        text = reader.pages[i].extract_text() or ""
        page_texts.append(text)

    full_text = ("\n\n" + PAGE_BREAK + "\n\n").join(page_texts)

    if clean:
        full_text = clean_text(full_text)

    return full_text


def _parse_pages_range(pages_range: str, total: int) -> tuple[int, int]:
    """Parse '10-50' into (10, 50). Clamps to valid range."""
    parts = pages_range.split("-")
    if len(parts) == 1:
        p = max(1, min(int(parts[0]), total))
        return p, p
    start = max(1, int(parts[0]))
    end = min(int(parts[1]), total)
    return start, end
