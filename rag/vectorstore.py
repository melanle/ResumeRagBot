"""Persistent Chroma vector store via LangChain.

Uses langchain_chroma.Chroma in local-persistent mode so the index survives
across sessions (stored under chroma_store/). Embeddings come from our
normalized Gemini wrapper; the collection uses cosine space to match.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .chunking import Chunk
from .config import CHROMA_DIR, settings
from .embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": settings.distance_space},
    )


def reset() -> None:
    """Drop all vectors (used on a fresh resume upload)."""
    get_store().reset_collection()


def add_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    docs = [
        Document(page_content=c.text, metadata={"section": c.section, **c.metadata})
        for c in chunks
    ]
    get_store().add_documents(documents=docs, ids=[c.id for c in chunks])


def count() -> int:
    return get_store()._collection.count()


def all_documents() -> list[Document]:
    """Every stored chunk as LangChain Documents (used to build BM25)."""
    if count() == 0:
        return []
    data = get_store().get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta, id=cid)
        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
