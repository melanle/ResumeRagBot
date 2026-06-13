"""End-to-end orchestration for the Resume RAG System.

ingest_resume(): files -> text -> structure/semantic chunks -> Chroma.
answer_query():  query -> hybrid retrieve -> rerank -> grounded generate
                 -> {answer, sources, confidence} + structured trace log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import observability as obs
from .chunking import chunk_resume
from .embeddings import embed_for_chunking
from .generate import generate_answer
from .ingest import extract_many
from .rerank import RankedChunk, rerank
from .retrieval import hybrid_retrieve
from . import vectorstore


@dataclass
class Source:
    tag: str
    section: str
    chunk_id: str
    relevance: float
    cited: bool
    text: str


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    confidence: float
    timings_ms: dict


def ingest_resume(file_paths: list[str], reset: bool = True) -> dict:
    """Process resume file(s) into the persistent vector store."""
    timings: dict = {}
    with obs.stage_timer(timings, "extract"):
        text = extract_many(file_paths)
    with obs.stage_timer(timings, "chunk"):
        chunks = chunk_resume(text, embed_fn=embed_for_chunking)
    if reset:
        vectorstore.reset()
    with obs.stage_timer(timings, "embed_store"):
        vectorstore.add_chunks(chunks)

    obs.log.info("ingested %d chunks across %d sections in %sms",
                 len(chunks), len({c.section for c in chunks}), timings)
    obs.log_event({
        "type": "ingest",
        "files": file_paths,
        "n_chunks": len(chunks),
        "sections": sorted({c.section for c in chunks}),
        "timings_ms": timings,
    })
    return {"n_chunks": len(chunks), "sections": sorted({c.section for c in chunks})}


def _confidence(reranked: list[RankedChunk], result_refused: bool, cited_ids: set[str]) -> float:
    """Confidence in [0,1] from reranker relevance and citation grounding.

    Blends the top chunk's relevance, the mean relevance of cited chunks, and
    a penalty when the model answered without citing anything. A refusal is
    reported as high confidence (the system is confident it lacks the info).
    """
    if result_refused:
        return round(0.9 if reranked else 0.6, 3)
    if not reranked:
        return 0.0
    top_rel = reranked[0].relevance
    cited = [r.relevance for r in reranked if r.chunk_id in cited_ids]
    mean_cited = sum(cited) / len(cited) if cited else 0.0
    citation_bonus = 1.0 if cited else 0.6  # uncited answer is less trustworthy
    score = (0.5 * top_rel + 0.5 * mean_cited) * citation_bonus
    return round(max(0.0, min(1.0, score)), 3)


def answer_query(query: str) -> Answer:
    timings: dict = {}

    with obs.stage_timer(timings, "retrieve"):
        candidates = hybrid_retrieve(query)
    with obs.stage_timer(timings, "rerank"):
        reranked = rerank(query, candidates)
    with obs.stage_timer(timings, "generate"):
        gen = generate_answer(query, reranked)

    cited_ids = set(gen.cited_source_ids)
    sources: list[dict] = []
    for i, ch in enumerate(reranked, start=1):
        sources.append(asdict(Source(
            tag=f"S{i}",
            section=ch.section,
            chunk_id=ch.chunk_id,
            relevance=ch.relevance,
            cited=ch.chunk_id in cited_ids,
            text=ch.text,
        )))

    confidence = _confidence(reranked, gen.refused, cited_ids)

    obs.log.info("query=%r -> confidence=%.2f, %d sources, timings=%s",
                 query, confidence, len(sources), timings)
    obs.log_event(obs.build_query_event(
        query=query,
        candidates=candidates,
        reranked=reranked,
        answer=gen.answer,
        sources=[{"tag": s["tag"], "section": s["section"], "cited": s["cited"],
                  "relevance": s["relevance"]} for s in sources],
        confidence=confidence,
        timings=timings,
    ))

    return Answer(answer=gen.answer, sources=sources, confidence=confidence, timings_ms=timings)


def is_indexed() -> bool:
    return vectorstore.count() > 0
