"""Resilient Gemini text-generation calls.

Wraps generate_content with retry + exponential backoff so transient quota
(429) and availability (503) errors don't abort a request or an evaluation
run. On a 429 the server often returns a suggested retry delay; we honor it
(capped) before falling back to exponential backoff.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache

import google.generativeai as genai
from google.api_core import exceptions as gax

from .config import settings


@lru_cache(maxsize=1)
def _configured() -> bool:
    genai.configure(api_key=settings.require_api_key())
    return True

MAX_RETRIES = 4
MAX_BACKOFF_S = 30.0


def _suggested_delay(exc: Exception) -> float | None:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"seconds:\s*(\d+)", str(exc))
    return float(match.group(1)) if match else None


def _is_daily_quota(exc: Exception) -> bool:
    """A per-day / hard quota (limit: 0) won't clear by waiting seconds."""
    text = str(exc)
    return "PerDay" in text or "limit: 0" in text


def generate_text(model_name: str, prompt: str, generation_config: dict | None = None) -> str:
    """Generate text, retrying on per-minute quota / transient errors.

    Fails fast (no backoff) on per-day quota exhaustion, which retrying within
    a few minutes cannot fix.
    """
    _configured()
    model = genai.GenerativeModel(model_name)
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = model.generate_content(prompt, generation_config=generation_config or {})
            return resp.text or ""
        except gax.ResourceExhausted as exc:
            last_exc = exc
            if _is_daily_quota(exc):
                raise RuntimeError(
                    f"Daily free-tier quota exhausted for {model_name}. Enable billing "
                    f"or use a key with generation quota. ({exc})"
                ) from exc
            if attempt == MAX_RETRIES - 1:
                break
            delay = min((_suggested_delay(exc) or (2.0 ** attempt) * 2.0) + 0.5, MAX_BACKOFF_S)
            print(f"[llm] rate-limited on {model_name}; retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s")
            time.sleep(delay)
        except (gax.ServiceUnavailable, gax.DeadlineExceeded, gax.InternalServerError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            delay = min((2.0 ** attempt) * 2.0 + 0.5, MAX_BACKOFF_S)
            print(f"[llm] {type(exc).__name__} on {model_name}; retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s")
            time.sleep(delay)

    raise RuntimeError(f"generate_text exhausted retries for {model_name}: {last_exc}")
