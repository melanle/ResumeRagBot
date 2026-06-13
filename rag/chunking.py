"""Structure-aware + semantic chunking for resumes.

Strategy (replaces the old fixed chunk_size=50 splitter):

1. Structure pass -- detect canonical resume sections (Summary, Skills,
   Experience, Projects, Education, ...) by header heuristics so every chunk
   carries a `section` label. This keeps skills/projects/experience content
   from bleeding into each other.

2. Semantic pass -- inside a section, group sentences and open a new chunk at
   *semantic breakpoints*: points where the cosine distance between adjacent
   sentence embeddings spikes above a percentile threshold. This yields
   topically-coherent chunks instead of arbitrary character windows.

The embedder is injected (a callable: list[str] -> np.ndarray of unit vectors)
so chunking stays testable and decoupled from the Gemini client. If no
embedder is supplied, the semantic pass degrades gracefully to size-based
grouping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import settings

EmbedFn = Callable[[list[str]], np.ndarray]

# Canonical section names + the header aliases that map onto them.
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Summary": ("summary", "objective", "profile", "about"),
    "Skills": ("skills", "technical skills", "core competencies", "technologies"),
    "Experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "work history",
    ),
    "Projects": ("projects", "personal projects", "selected projects"),
    "Education": ("education", "academic background"),
    "Certifications": ("certifications", "certificates", "licenses"),
    "Awards": ("awards", "honors", "achievements"),
    "Publications": ("publications", "papers"),
    "Languages": ("languages",),
    "Contact": ("contact", "contact information"),
}


@dataclass
class Chunk:
    text: str
    section: str
    index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.section.lower()}-{self.index}"


def _classify_header(line: str) -> Optional[str]:
    """Return the canonical section name if `line` looks like a header."""
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return None
    # Headers are short, often all-caps or title-case, and rarely end with a period.
    normalized = re.sub(r"[^a-z ]", "", stripped.lower()).strip()
    if not normalized:
        return None
    for canonical, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split raw resume text into (section_name, section_text) pairs."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    # Everything before the first recognized header is the header/contact block.
    current_name = "Header"
    current_lines: list[str] = []

    for line in lines:
        header = _classify_header(line)
        if header is not None:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = header
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_name, current_lines))

    return [(name, "\n".join(ls).strip()) for name, ls in sections if "\n".join(ls).strip()]


def _split_sentences(text: str) -> list[str]:
    """Sentence/line segmentation that respects resume bullet structure."""
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Bullet lines are atomic units; prose lines split on sentence ends.
        if re.match(r"^[\-\*•]", line):
            units.append(re.sub(r"^[\-\*•]\s*", "", line))
        else:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
            units.extend(p.strip() for p in parts if p.strip())
    return units


def _semantic_breakpoints(units: list[str], embed_fn: EmbedFn) -> list[int]:
    """Indices in `units` after which a new chunk should begin."""
    if len(units) < 2:
        return []
    vecs = embed_fn(units)
    # Cosine distance between consecutive (already unit-normalized) sentences.
    dists = [1.0 - float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(units) - 1)]
    if not dists:
        return []
    threshold = float(np.percentile(dists, settings.semantic_breakpoint_percentile))
    return [i + 1 for i, d in enumerate(dists) if d >= threshold and d > 0]


def _group_units(
    units: list[str], breakpoints: set[int]
) -> list[str]:
    """Greedily merge units into chunks, honoring breakpoints and size caps."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(" ".join(buf).strip())
            buf, buf_len = [], 0

    for i, unit in enumerate(units):
        unit_len = len(unit)
        starts_new = i in breakpoints and buf_len >= settings.chunk_target_chars // 2
        too_big = buf_len + unit_len > settings.chunk_max_chars
        if buf and (starts_new or too_big):
            flush()
        buf.append(unit)
        buf_len += unit_len + 1
        if buf_len >= settings.chunk_target_chars and i in breakpoints:
            flush()
    flush()
    return [c for c in chunks if c]


def _apply_overlap(chunks: list[str]) -> list[str]:
    """Prepend a short tail of the previous chunk to preserve context."""
    overlap = settings.chunk_overlap_chars
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        # Snap to a word boundary so we don't slice mid-token.
        if " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        out.append(f"{tail} {cur}".strip())
    return out


def chunk_resume(text: str, embed_fn: Optional[EmbedFn] = None) -> list[Chunk]:
    """Produce structure-aware, semantically-grouped chunks from resume text."""
    chunks: list[Chunk] = []
    running_index = 0

    for section_name, section_text in split_into_sections(text):
        units = _split_sentences(section_text)
        if not units:
            continue

        if embed_fn is not None and len(units) > 2:
            breakpoints = set(_semantic_breakpoints(units, embed_fn))
        else:
            breakpoints = set(range(len(units)))  # size-only grouping fallback

        grouped = _group_units(units, breakpoints)
        grouped = _apply_overlap(grouped)

        for piece in grouped:
            # Carry the section header into the text so embeddings and the LLM
            # both see which part of the resume a chunk came from.
            body = f"[{section_name}] {piece}"
            chunks.append(
                Chunk(
                    text=body,
                    section=section_name,
                    index=running_index,
                    metadata={"section": section_name, "char_len": len(piece)},
                )
            )
            running_index += 1

    return chunks
