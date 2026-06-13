"""Hybrid retrieval via LangChain's EnsembleRetriever + dedup.

Dense (Chroma vector) and sparse (BM25) retrievers are combined with
EnsembleRetriever, which fuses their ranked lists using Reciprocal Rank
Fusion -- robust to the different score scales of cosine vs BM25. We then
collapse near-duplicate chunks (overlapping windows) with token Jaccard so the
reranker/LLM don't see the same content twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# LangChain v1 moved the legacy ensemble retriever into langchain_classic.
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import settings
from .vectorstore import all_documents, count, get_store


@dataclass
class RetrievedChunk:
    id: str
    text: str
    section: str
    rank: int
    score: float                       # RRF-rank pseudo-score (1 / (rank+1))
    sources: list[str] = field(default_factory=lambda: ["hybrid"])


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _doc_id(doc: Document, fallback: int) -> str:
    return doc.id or doc.metadata.get("section", "chunk") + f"-{fallback}"


def hybrid_retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return fused, deduplicated chunks ranked by RRF (desc)."""
    top_k = top_k or settings.retrieval_top_k
    if count() == 0:
        return []

    docs = all_documents()
    vector_retriever = get_store().as_retriever(search_kwargs={"k": min(top_k, len(docs))})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = min(top_k, len(docs))

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    fused = ensemble.invoke(query)

    candidates = [
        RetrievedChunk(
            id=_doc_id(doc, rank),
            text=doc.page_content,
            section=doc.metadata.get("section", "?"),
            rank=rank,
            score=round(1.0 / (rank + 1), 5),
        )
        for rank, doc in enumerate(fused)
    ]

    # Near-duplicate removal (keeps the higher-ranked of any overlapping pair).
    kept: list[RetrievedChunk] = []
    for cand in candidates:
        if any(_jaccard(cand.text, k.text) >= settings.dedup_jaccard_threshold for k in kept):
            continue
        kept.append(cand)
    return kept[:top_k]
