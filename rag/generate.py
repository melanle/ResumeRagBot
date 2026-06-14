"""Grounded, citation-based answer generation.

The prompt forces the model to answer ONLY from the retrieved passages, cite
each claim with the bracketed source id, and explicitly refuse when the
passages don't contain the answer. This is the primary anti-hallucination
control; faithfulness is then measured independently in evaluate.py.
"""

from __future__ import annotations

import re
from datetime import date
from dataclasses import dataclass

from .config import settings
from .llm import generate_text
from .rerank import RankedChunk

_SYSTEM_PROMPT = """You are a Career Intelligence assistant that answers questions
about a candidate STRICTLY from the resume passages provided below.

Today's date is {today}.

RULES (follow exactly):
1. Use ONLY information present in the SOURCES. Do not use outside knowledge.
2. Cite every factual claim with its source tag, e.g. "led a team of 6 [S2]".
3. If the SOURCES do not contain the answer, reply EXACTLY:
   "No information regarding this in the resume."
   Do NOT guess, infer beyond the text, or fabricate dates, numbers, or names.
4. DATE CALCULATIONS: If a date range in the sources is open-ended ("Present",
   "Current", "Now"), treat its end as today's date ({today}). To answer total
   years/months of experience, find the employment date ranges in the sources,
   compute each duration, and report the total (approximate is fine). Briefly
   show the dates you used, e.g. "~1 year (May 2025-Present) [S1]". This counts
   as information present in the resume -- do NOT refuse just because no explicit
   "X years" sentence exists, as long as employment dates are given.
5. Be concise and factual.

SOURCES:
{sources}

QUESTION: {question}

Answer (with [S#] citations):"""


@dataclass
class GenerationResult:
    answer: str
    cited_source_ids: list[str]
    refused: bool


def _format_sources(chunks: list[RankedChunk]) -> tuple[str, dict[str, RankedChunk]]:
    blocks = []
    tag_map: dict[str, RankedChunk] = {}
    for i, ch in enumerate(chunks, start=1):
        tag = f"S{i}"
        tag_map[tag] = ch
        blocks.append(f"[{tag}] (section: {ch.section}) {ch.text}")
    return "\n\n".join(blocks), tag_map


def generate_answer(question: str, chunks: list[RankedChunk]) -> GenerationResult:
    if not chunks:
        return GenerationResult(
            answer="No information regarding this in the resume.",
            cited_source_ids=[],
            refused=True,
        )

    sources_block, tag_map = _format_sources(chunks)
    prompt = _SYSTEM_PROMPT.format(
        sources=sources_block, question=question, today=date.today().isoformat()
    )

    answer = generate_text(
        settings.generation_model,
        prompt,
        generation_config={"temperature": settings.generation_temperature},
    ).strip()

    cited_tags = set(re.findall(r"\[(S\d+)\]", answer))
    cited_source_ids = [tag_map[t].chunk_id for t in cited_tags if t in tag_map]
    refused = answer.lower().startswith("no information regarding this")

    return GenerationResult(
        answer=answer,
        cited_source_ids=cited_source_ids,
        refused=refused,
    )
