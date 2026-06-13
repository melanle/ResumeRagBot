"""LLM-based reranking layer (cross-encoder-style relevance scoring).

The first-stage hybrid retriever optimizes recall (15 candidates). A true
cross-encoder reranker would jointly encode (query, chunk) pairs; here we use
Gemini as the cross-encoder: it scores each candidate's relevance to the query
on a 0-10 scale in a single batched call, then we keep the top N (3-5).

Why LLM instead of a local cross-encoder model: this project targets Python
3.14 where torch/sentence-transformers wheels are not yet available, and it
keeps the entire stack on one provider (no extra model download). The
RerankScorer interface is deliberately swappable -- a sentence-transformers
CrossEncoder could drop in behind the same `rerank()` signature.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import settings
from .llm import generate_text
from .retrieval import RetrievedChunk


@dataclass
class RankedChunk:
    chunk_id: str
    section: str
    text: str
    relevance: float          # 0..1 (rerank score / 10)
    retrieval_sources: list[str]


_RERANK_PROMPT = """You are a precise relevance judge for a resume search system.
Score how well each numbered passage helps answer the QUESTION, on a 0-10 scale:
  10 = directly and fully answers the question
  5  = partially relevant / related context
  0  = irrelevant
Judge ONLY relevance to the question, not writing quality.

QUESTION: {query}

PASSAGES:
{passages}

Return ONLY a JSON array of objects: [{{"id": <passage number>, "score": <0-10>}}].
No prose, no markdown fences."""


def _parse_scores(raw: str, n: int) -> dict[int, float]:
    # Strip accidental code fences and isolate the JSON array.
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    scores: dict[int, float] = {}
    for item in data:
        try:
            idx = int(item["id"])
            score = max(0.0, min(10.0, float(item["score"])))
            if 0 <= idx < n:
                scores[idx] = score
        except (KeyError, ValueError, TypeError):
            continue
    return scores


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int | None = None) -> list[RankedChunk]:
    top_n = top_n or settings.rerank_top_n
    if not candidates:
        return []

    passages = "\n\n".join(
        f"[{i}] {c.text}" for i, c in enumerate(candidates)
    )
    prompt = _RERANK_PROMPT.format(query=query, passages=passages)

    try:
        raw = generate_text(
            settings.rerank_model,
            prompt,
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        scores = _parse_scores(raw, len(candidates))
    except Exception:
        scores = {}

    # Fallback: if the LLM scoring fails entirely, keep fusion order.
    if not scores:
        scores = {i: 10.0 * (1.0 - i / max(1, len(candidates))) for i in range(len(candidates))}

    ranked = sorted(
        range(len(candidates)),
        key=lambda i: (scores.get(i, 0.0), candidates[i].score),
        reverse=True,
    )

    out: list[RankedChunk] = []
    for i in ranked[:top_n]:
        cand = candidates[i]
        out.append(
            RankedChunk(
                chunk_id=cand.id,
                section=cand.section,
                text=cand.text,
                relevance=round(scores.get(i, 0.0) / 10.0, 4),
                retrieval_sources=cand.sources,
            )
        )
    return out
