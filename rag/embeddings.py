"""Gemini embeddings (LangChain) with L2 normalization.

Wraps LangChain's GoogleGenerativeAIEmbeddings (model: gemini-embedding-001,
a stronger model than the legacy embedding-001) and L2-normalizes every
vector before it leaves this module, so cosine similarity in Chroma reduces to
a dot product. LangChain automatically uses the RETRIEVAL_DOCUMENT task type
for documents and RETRIEVAL_QUERY for queries, which improves retrieval.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import settings


def _normalize_one(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr)) or 1.0
    return (arr / norm).tolist()


class NormalizedGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """GoogleGenerativeAIEmbeddings that returns unit-normalized vectors."""

    def embed_documents(self, texts: list[str], *args, **kwargs) -> list[list[float]]:
        return [_normalize_one(v) for v in super().embed_documents(texts, *args, **kwargs)]

    def embed_query(self, text: str, *args, **kwargs) -> list[float]:
        return _normalize_one(super().embed_query(text, *args, **kwargs))


@lru_cache(maxsize=1)
def get_embeddings() -> NormalizedGeminiEmbeddings:
    return NormalizedGeminiEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.require_api_key(),
    )


def embed_for_chunking(texts: list[str]) -> np.ndarray:
    """Unit-normalized embeddings as a matrix, for the semantic chunker."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return np.asarray(get_embeddings().embed_documents(texts), dtype=np.float32)
