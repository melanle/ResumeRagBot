"""Structured observability for the RAG pipeline.

Every query emits one JSON line to logs/rag_events.jsonl capturing the full
trace: the query, the retrieved candidates, the reranked chunks (with scores),
the final answer, confidence, and stage latencies. This makes retrieval
quality debuggable after the fact and feeds dashboards / evaluation.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any

from .config import settings

# Human-readable console logger.
_console = logging.getLogger("rag")
if not _console.handlers:
    _console.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _console.addHandler(handler)

log = _console


def _monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


@contextmanager
def stage_timer(store: dict, key: str):
    start = _monotonic_ms()
    try:
        yield
    finally:
        store[key] = round(_monotonic_ms() - start, 1)


def log_event(event: dict[str, Any]) -> None:
    """Append one structured event to the JSONL trace file."""
    try:
        with open(settings.log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # logging must never crash the request
        log.warning("failed to write trace event: %s", exc)


def build_query_event(
    *,
    query: str,
    candidates,
    reranked,
    answer: str,
    sources,
    confidence: float,
    timings: dict,
) -> dict[str, Any]:
    return {
        "type": "query",
        "query": query,
        "retrieved_chunks": [
            {
                "id": c.id,
                "section": c.section,
                "rank": c.rank,
                "rrf_score": c.score,
                "sources": c.sources,
                "preview": c.text[:120],
            }
            for c in candidates
        ],
        "reranked_chunks": [
            {
                "id": r.chunk_id,
                "section": r.section,
                "relevance": r.relevance,
                "preview": r.text[:120],
            }
            for r in reranked
        ],
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "timings_ms": timings,
    }
