"""Evaluation harness for the Resume RAG System.

Runs the full retrieval pipeline over evaluation/test_questions.json and
reports three families of metrics:

  * Retrieval precision@k   -- fraction of top-k retrieved/reranked chunks whose
                               resume section matches the annotated relevant
                               section(s). Measured on both the first-stage
                               hybrid candidates and the post-rerank chunks.
  * Answer faithfulness     -- LLM judge scores whether every claim in the
                               answer is supported by the retrieved passages
                               (the anti-hallucination metric).
  * Context relevance       -- LLM judge scores whether the retrieved context
                               is actually about the question.

It also reports refusal accuracy on the unanswerable questions.

Usage:
    python evaluate.py                       # uses data/sample_resume.txt
    python evaluate.py --resume path.pdf     # evaluate against another resume
    python evaluate.py --k 5 --no-judge      # precision only, skip LLM judging
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from rag.config import BASE_DIR, settings
from rag.llm import generate_text
from rag.generate import generate_answer
from rag.pipeline import ingest_resume
from rag.rerank import rerank
from rag.retrieval import hybrid_retrieve

DATASET_PATH = BASE_DIR / "evaluation" / "test_questions.json"


# --------------------------------------------------------------------------
# LLM judges
# --------------------------------------------------------------------------
def _judge_json(prompt: str) -> dict:
    try:
        raw = generate_text(
            settings.judge_model,
            prompt,
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        text = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}


def judge_faithfulness(question: str, answer: str, passages: list[str]) -> float:
    if not answer.strip():
        return 0.0
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = f"""You are a strict faithfulness judge. Decide whether the ANSWER is
fully supported by the CONTEXT passages (no fabricated facts, numbers, or names).
A refusal ("No information regarding this in the resume.") is fully faithful when
the context indeed lacks the answer.

CONTEXT:
{ctx}

QUESTION: {question}
ANSWER: {answer}

Return JSON: {{"faithfulness": <float 0..1>, "unsupported_claims": [<strings>]}}"""
    result = _judge_json(prompt)
    try:
        return max(0.0, min(1.0, float(result.get("faithfulness", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def judge_context_relevance(question: str, passages: list[str]) -> float:
    if not passages:
        return 0.0
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = f"""Rate how relevant the retrieved CONTEXT is to the QUESTION overall,
on a 0..1 scale (1 = every passage is on-topic, 0 = none are).

CONTEXT:
{ctx}

QUESTION: {question}

Return JSON: {{"context_relevance": <float 0..1>}}"""
    result = _judge_json(prompt)
    try:
        return max(0.0, min(1.0, float(result.get("context_relevance", 0.0))))
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def precision_at_k(sections: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks whose section is relevant.

    Note: when an answer lives in a single section, precision@k is capped at
    1/k by construction, so read it alongside hit@k / recall@k below.
    """
    if not relevant or k <= 0:
        return float("nan")  # not applicable (e.g. refusal questions)
    topk = sections[:k]
    if not topk:
        return 0.0
    hits = sum(1 for s in topk if s in relevant)
    return hits / len(topk)


def hit_at_k(sections: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant section appears in the top-k, else 0.0 (recall-ish)."""
    if not relevant or k <= 0:
        return float("nan")
    return 1.0 if any(s in relevant for s in sections[:k]) else 0.0


def recall_at_k(sections: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the distinct relevant sections that appear in the top-k."""
    if not relevant or k <= 0:
        return float("nan")
    found = {s for s in sections[:k] if s in relevant}
    return len(found) / len(relevant)


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return round(statistics.fmean(vals), 4) if vals else float("nan")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run(resume_path: str | None, k: int, use_judge: bool) -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    resume = resume_path or str(BASE_DIR / dataset["resume_file"])

    print(f"Indexing resume: {resume}")
    info = ingest_resume([resume], reset=True)
    print(f"  -> {info['n_chunks']} chunks, sections: {', '.join(info['sections'])}\n")

    per_question = []
    for item in dataset["questions"]:
        q = item["question"]
        relevant = set(item.get("relevant_sections", []))
        expected_refusal = item.get("expected_refusal", False)

        try:
            candidates = hybrid_retrieve(q)
            reranked = rerank(q, candidates)
            gen = generate_answer(q, reranked)
        except RuntimeError as exc:
            # e.g. daily quota exhausted mid-run; save what we have and stop.
            print(f"\n  [STOP] {item['id']}: {exc}")
            print(f"  Aborting after {len(per_question)} completed questions; writing partial report.")
            break

        cand_sections = [c.section for c in candidates]
        rerank_sections = [r.section for r in reranked]
        passages = [r.text for r in reranked]

        row = {
            "id": item["id"],
            "question": q,
            "answer": gen.answer,
            "refused": gen.refused,
            "expected_refusal": expected_refusal,
            "retrieval_precision@{}".format(k): precision_at_k(cand_sections, relevant, k),
            "retrieval_hit@{}".format(k): hit_at_k(cand_sections, relevant, k),
            "retrieval_recall@{}".format(k): recall_at_k(cand_sections, relevant, k),
            "rerank_precision@{}".format(len(reranked) or 1): precision_at_k(
                rerank_sections, relevant, len(reranked) or 1
            ),
            "refusal_correct": (gen.refused == expected_refusal),
            "rerank_sections": rerank_sections,
        }
        if use_judge:
            row["faithfulness"] = judge_faithfulness(q, gen.answer, passages)
            row["context_relevance"] = judge_context_relevance(q, passages)

        per_question.append(row)
        flag = "OK " if row["refusal_correct"] else "!! "
        print(f"  [{flag}] {item['id']}: P@{k}={row['retrieval_precision@'+str(k)]!s:<5} "
              + (f"faith={row.get('faithfulness')!s:<5} ctx={row.get('context_relevance')!s:<5} " if use_judge else "")
              + f"{q[:48]}")

    # Aggregate ------------------------------------------------------------
    if not per_question:
        print("\nNo questions completed (generation quota unavailable). "
              "Enable billing or use a key with quota, then re-run.")
        return {"summary": {}, "per_question": []}

    pk_key = f"retrieval_precision@{k}"
    rerank_keys = [key for key in per_question[0] if key.startswith("rerank_precision@")]
    rerank_key = rerank_keys[0] if rerank_keys else None

    summary = {
        "n_questions": len(per_question),
        f"mean_retrieval_precision@{k}": _mean([r[pk_key] for r in per_question]),
        f"mean_retrieval_hit@{k}": _mean([r[f"retrieval_hit@{k}"] for r in per_question]),
        f"mean_retrieval_recall@{k}": _mean([r[f"retrieval_recall@{k}"] for r in per_question]),
        "refusal_accuracy": round(
            sum(r["refusal_correct"] for r in per_question) / len(per_question), 4
        ),
    }
    if rerank_key:
        summary[f"mean_{rerank_key}"] = _mean([r[rerank_key] for r in per_question])
    if use_judge:
        summary["mean_faithfulness"] = _mean([r["faithfulness"] for r in per_question])
        summary["mean_context_relevance"] = _mean([r["context_relevance"] for r in per_question])

    report = {"summary": summary, "per_question": per_question}

    out_path = BASE_DIR / "evaluation" / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, val in summary.items():
        print(f"  {key:<36} {val}")
    print(f"\nFull report written to {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Resume RAG System")
    parser.add_argument("--resume", help="Path to a resume file (default: sample resume)")
    parser.add_argument("--k", type=int, default=5, help="k for precision@k (default 5)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM faithfulness/relevance judging")
    args = parser.parse_args()
    settings.require_api_key()
    run(resume_path=args.resume, k=args.k, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
