# Resume RAG System

A production-grade retrieval-augmented generation system for resumes. It goes
well beyond naive "PDF Q&A": a persistent vector database, structure-aware
semantic chunking, hybrid (dense + keyword) retrieval, an LLM reranking layer,
grounded citation-based answers, full observability, and a reproducible
evaluation harness.

This is an upgrade of an earlier FAISS + `chunk_size=50` prototype.

## Architecture

```
upload ─► ingest (PDF/TXT) ─► structure-aware + semantic chunking
                                        │
                                        ▼
                      normalized Gemini embeddings (gemini-embedding-001)
                                        │
                                        ▼
                          persistent Chroma vector store
                                        │
 query ─►  ┌───────────────────────────┴───────────────────────────┐
           │  HYBRID RETRIEVAL                                       │
           │  dense (cosine)  +  BM25 keyword  →  Reciprocal Rank    │
           │  Fusion  →  near-duplicate dedup  →  top 15 candidates  │
           └───────────────────────────┬───────────────────────────┘
                                        ▼
                LLM reranker (cross-encoder-style 0–10 scoring) → top 4
                                        ▼
            grounded generation (citations + explicit refusal on miss)
                                        ▼
              { answer, sources (sections used), confidence }
                                        ▼
                  structured JSONL trace → logs/rag_events.jsonl
```

### How each requirement is met

| Requirement                         | Implementation                                                                                                                                                               |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Real vector DB, persistent          | `langchain_chroma.Chroma` in [rag/vectorstore.py](rag/vectorstore.py); persists across sessions in `chroma_store/`                                                           |
| Semantic + structural chunking      | [rag/chunking.py](rag/chunking.py): section detection (Skills/Experience/Projects/…) + embedding-based semantic breakpoints, replacing fixed `chunk_size=50`                 |
| Reranking layer                     | [rag/rerank.py](rag/rerank.py): Gemini scores 15 candidates 0–10, returns top 4                                                                                              |
| Hybrid retrieval + dedup            | [rag/retrieval.py](rag/retrieval.py): LangChain `EnsembleRetriever` fuses dense + BM25 via RRF, then Jaccard near-dup removal                                                |
| Grounded, anti-hallucination prompt | [rag/generate.py](rag/generate.py): cite-or-refuse system prompt                                                                                                             |
| Stronger embeddings, normalized     | [rag/embeddings.py](rag/embeddings.py): LangChain `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`), L2-normalized, task-typed query/document                          |
| Resilient LLM calls                 | [rag/llm.py](rag/llm.py): retry + exponential backoff on 429/503 so free-tier rate limits don't abort a run                                                                  |
| Evaluation framework                | [evaluate.py](evaluate.py) + [evaluation/test_questions.json](evaluation/test_questions.json) (23 questions): precision@k, faithfulness, context relevance, refusal accuracy |
| Observability                       | [rag/observability.py](rag/observability.py): logs query, retrieved chunks, reranked chunks, answer, confidence, latencies                                                   |
| UX response                         | answer + sources (sections) + confidence score, in both the UI and `/api/ask`                                                                                                |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your GOOGLE_API_KEY
```

Get a key from Google AI Studio (https://aistudio.google.com/app/apikey).

## Run the app

```bash
python app.py
# open http://127.0.0.1:5000
```

Upload a resume (PDF), then ask questions. Each answer shows the resume
sections used, which were cited, and a confidence score.

### JSON API

```bash
curl -s http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which vector databases has the candidate used?"}'
```

## Run the evaluation

```bash
python evaluate.py                 # full metrics on the bundled sample resume
python evaluate.py --k 5           # precision@5
python evaluate.py --no-judge      # skip LLM judging (precision + refusal only)
```

Writes a detailed `evaluation/eval_report.json` and prints a summary:
mean retrieval precision@k, mean rerank precision, mean faithfulness,
mean context relevance, and refusal accuracy.

The bundled [data/sample_resume.txt](data/sample_resume.txt) makes evaluation
fully reproducible without shipping a binary PDF.

## Configuration

All knobs (models, chunk sizes, `top_k`, `rerank_top_n`, RRF constant, dedup
threshold) are centralized in [rag/config.py](rag/config.py) and overridable
via environment variables — see [.env.example](.env.example).

## Notes / design choices

- **Reranker is LLM-based**, not a local cross-encoder. Target runtime is
  Python 3.14, where `torch`/`sentence-transformers` wheels aren't yet
  available. The `rerank()` interface is swappable — a `CrossEncoder` can drop
  in behind the same signature on a supported Python.
- **One provider (Gemini)** end to end (embeddings, generation, reranking,
  judging) so no extra API keys or model downloads are needed.
- **LangChain is used for the heavy plumbing** — `Chroma` vector store,
  `GoogleGenerativeAIEmbeddings`, and `EnsembleRetriever` (which provides RRF).
  The custom layers (semantic/structural chunking, LLM reranking, confidence
  scoring, observability, retry, evaluation) are kept first-party so the
  retrieval engineering stays explicit and debuggable. Note LangChain v1 moved
  `EnsembleRetriever` into the `langchain_classic` package.
