"""Flask front-end for the Resume RAG System.

Thin presentation layer over rag.pipeline. Uploading a resume re-indexes the
persistent Chroma store; asking a question returns a grounded answer plus the
resume sections used and a confidence score.
"""

import os

from flask import Flask, jsonify, render_template, request

from rag.config import UPLOAD_DIR
from rag import pipeline

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


def _current_resume() -> str | None:
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    return files[0] if files else None


def _save_uploads(file_storage_list) -> list[str]:
    # Clear previous uploads so the active resume is unambiguous.
    for name in os.listdir(app.config["UPLOAD_FOLDER"]):
        path = os.path.join(app.config["UPLOAD_FOLDER"], name)
        if os.path.isfile(path):
            os.remove(path)
    saved = []
    for f in file_storage_list:
        if not f.filename:
            continue
        path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        f.save(path)
        saved.append(path)
    return saved


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        if request.files.getlist("pdf") and any(f.filename for f in request.files.getlist("pdf")):
            saved = _save_uploads(request.files.getlist("pdf"))
            if saved:
                pipeline.ingest_resume(saved, reset=True)

        question = request.form.get("question", "").strip()
        if question:
            if not pipeline.is_indexed():
                error = "Please upload a resume before asking a question."
            else:
                result = pipeline.answer_query(question)

    return render_template(
        "home.html",
        result=result,
        error=error,
        uploaded_resume=_current_resume(),
        indexed=pipeline.is_indexed(),
    )


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """JSON API: {"question": "..."} -> {answer, sources, confidence}."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    if not pipeline.is_indexed():
        return jsonify({"error": "no resume indexed"}), 409
    ans = pipeline.answer_query(question)
    return jsonify({
        "answer": ans.answer,
        "confidence": ans.confidence,
        "sources": [
            {"tag": s["tag"], "section": s["section"], "relevance": s["relevance"], "cited": s["cited"]}
            for s in ans.sources
        ],
        "timings_ms": ans.timings_ms,
    })


if __name__ == "__main__":
    app.run(debug=True)
