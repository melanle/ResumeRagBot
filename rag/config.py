"""Central configuration for the Resume RAG System.

All tunable knobs live here so the pipeline, the Flask app, and the
evaluation harness share one source of truth. Values can be overridden
through environment variables (loaded from a .env file if present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project layout -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"      # persistent vector DB
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"

for _d in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- API / models -----------------------------------------------------
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    # Stronger embedding model than the legacy models/embedding-001.
    # gemini-embedding-001 is Google's current top embedding model.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    # gemini-2.5-flash is the model with usable free-tier quota on this key
    # (flash-lite / 2.0-flash return limit:0). Override per .env if you enable
    # billing and want to split rerank/judge onto a cheaper model.
    generation_model: str = os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")
    rerank_model: str = os.getenv("RERANK_MODEL", "models/gemini-2.5-flash")
    judge_model: str = os.getenv("JUDGE_MODEL", "models/gemini-2.5-flash")

    # --- Vector store -----------------------------------------------------
    collection_name: str = os.getenv("CHROMA_COLLECTION", "resume_chunks")
    # cosine works hand-in-hand with the L2 normalization in embeddings.py.
    distance_space: str = "cosine"

    # --- Chunking ---------------------------------------------------------
    # Target chunk size in characters; semantic splitter aims near this.
    chunk_target_chars: int = _get_int("CHUNK_TARGET_CHARS", 700)
    chunk_max_chars: int = _get_int("CHUNK_MAX_CHARS", 1100)
    chunk_overlap_chars: int = _get_int("CHUNK_OVERLAP_CHARS", 120)
    # Cosine-distance percentile at which a semantic breakpoint is created.
    semantic_breakpoint_percentile: float = _get_float(
        "SEMANTIC_BREAKPOINT_PERCENTILE", 80.0
    )

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = _get_int("RETRIEVAL_TOP_K", 15)   # candidates before rerank
    rerank_top_n: int = _get_int("RERANK_TOP_N", 4)          # chunks sent to the LLM
    rrf_k: int = _get_int("RRF_K", 60)                       # reciprocal-rank-fusion constant
    dedup_jaccard_threshold: float = _get_float("DEDUP_JACCARD_THRESHOLD", 0.85)

    # --- Generation -------------------------------------------------------
    generation_temperature: float = _get_float("GENERATION_TEMPERATURE", 0.2)

    # --- Observability ----------------------------------------------------
    log_file: Path = field(default_factory=lambda: LOG_DIR / "rag_events.jsonl")

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
                "with GOOGLE_API_KEY=<your key> before running the pipeline."
            )
        return self.google_api_key


settings = Settings()
