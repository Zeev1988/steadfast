"""
Stage 3: LLM-based ticket classification.

Prompt copy lives in ``agent.prompts``; other modules here wire LiteLLM, retries,
and parsing only.

Design decisions
----------------
- Provider abstraction: LiteLLM — a single ``acompletion()`` call works with
  any provider.  Model id is **only** from ``LLM_MODEL`` env (default
  ``claude-sonnet-4-20250514``), e.g. ``gpt-4o``, ``mistral/mistral-large-latest``.
  LiteLLM reads the matching API key env var automatically
  (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.).
- Context: BM25 top-5 KB chunks injected into the system message so the model
  can ground its classification and response in Steadfast-specific terminology,
  known issues, and historical resolutions.
- Output: JSON requested via explicit schema in the prompt.  A lightweight
  regex/json parser extracts the first JSON object from the response.
- Concurrency: asyncio.gather with a semaphore (default 5) keeps throughput
  high without hammering rate limits.
- Retries: Tenacity-based.  Rate-limit (429) and server errors (5xx) use
  exponential backoff with jitter.  Parse errors retry immediately.
  Non-retryable errors (401, 400) fail fast.
- Fallback: on total failure, return category="unknown", priority="medium"
  with an "llm_failure" flag so downstream stages can handle gracefully.
- Chain-of-thought: the prompt asks for a "reasoning" field which improves
  classification accuracy; it is stripped from the final output.
"""

from __future__ import annotations

from .classify import classify_tickets, classify_tickets_sync
from .config import MODEL

__all__ = ["MODEL", "classify_tickets", "classify_tickets_sync"]
