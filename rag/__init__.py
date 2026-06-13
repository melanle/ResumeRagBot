"""Resume RAG System.

A production-grade retrieval pipeline over resume documents:
ingestion -> structure-aware/semantic chunking -> normalized Gemini
embeddings -> persistent Chroma store -> hybrid (vector + BM25) retrieval
-> LLM reranking -> grounded, citation-based generation.
"""

from .config import settings

__all__ = ["settings"]
