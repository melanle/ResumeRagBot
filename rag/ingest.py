"""Document ingestion: turn resume files into clean plain text.

Supports PDF (via PyMuPDF) and plain-text resumes. Plain text is handy for
reproducible evaluation (see data/sample_resume.txt) where shipping a binary
PDF would be awkward.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


def _normalize_whitespace(text: str) -> str:
    # Collapse runs of spaces/tabs but preserve line breaks, which carry
    # structural meaning in resumes (section headers, bullet boundaries).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: str | Path) -> str:
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return _normalize_whitespace("\n".join(parts))


def extract_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return _normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
    raise ValueError(f"Unsupported resume file type: {path.suffix!r}")


def extract_many(paths: Iterable[str | Path]) -> str:
    """Concatenate text from multiple files (e.g. multi-page uploads)."""
    return _normalize_whitespace("\n\n".join(extract_text(p) for p in paths))
