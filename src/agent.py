"""
Stage 3: LLM-based ticket classification.

Prompt copy lives in ``triage_prompts.py``; this file wires LiteLLM, retries,
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

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

import litellm
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import ValidationError
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from models import LlmTriagePayload, Ticket, TriageResult
from triage_prompts import (
    CATEGORY_DESCRIPTIONS,
    PRIORITY_DESCRIPTIONS,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

# Silence litellm's verbose default logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Optional: if the user has a custom base URL (e.g. a proxy), set it globally.
_BASE_URL = os.environ.get("LLM_BASE_URL")
if _BASE_URL:
    litellm.api_base = _BASE_URL

MODEL = os.environ.get("LLM_MODEL", None)
if not MODEL:
    raise ValueError("LLM_MODEL environment variable is not set")

MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Prompt assembly (templates live in triage_prompts.py)
# ---------------------------------------------------------------------------


def _build_system_prompt(kb_chunks: list[str]) -> str:
    """Assemble the full system prompt with KB context."""
    if kb_chunks:
        context_block = "\n\n---\n\n".join(
            f"### Similar Ticket {i+1}\n{chunk}"
            for i, chunk in enumerate(kb_chunks)
        )
    else:
        context_block = "(No similar tickets found in the knowledge base.)"

    return SYSTEM_PROMPT.format(
        categories=CATEGORY_DESCRIPTIONS,
        priorities=PRIORITY_DESCRIPTIONS,
        kb_context=context_block,
    )


def _build_user_message(ticket: Ticket) -> str:
    """Format the incoming ticket as the user message."""
    parts = [
        f"Ticket ID: {ticket.ticket_id}",
        f"Customer: {ticket.customer_name}" if ticket.customer_name else None,
        f"Plan: {ticket.plan}" if ticket.plan else None,
        f"Subject: {ticket.subject}",
        f"Body:\n{ticket.body}",
    ]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Match the first { ... } block (greedy, handles nested braces one level deep)
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _parse_response(raw_text: str, ticket_id: str) -> TriageResult:
    """Extract JSON from the LLM response and build a TriageResult.

    Raises ValueError on parse failure so the caller can retry.
    """
    match = _JSON_RE.search(raw_text)
    if not match:
        raise ValueError(f"No JSON object found in LLM response for {ticket_id}")

    data = json.loads(match.group())

    try:
        payload = LlmTriagePayload.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM JSON schema invalid for {ticket_id}: {exc}") from exc

    category = payload.category.strip().lower()
    priority = payload.priority.strip().lower()

    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        response=payload.response,
        confidence=payload.confidence,
    )


# ---------------------------------------------------------------------------
# LLM call with retries (Tenacity + LiteLLM)
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception warrants a retry.

    Retryable:
      - Rate-limit (429) and server errors (5xx) — surfaced by LiteLLM as
        litellm.RateLimitError, litellm.ServiceUnavailableError, or
        litellm.InternalServerError.
      - Transient network errors (ConnectionError, TimeoutError).
      - JSON parse / validation errors (ValueError) — the LLM might produce
        valid output on the next attempt.

    Non-retryable:
      - Authentication errors (401), bad-request (400), not-found (404)
        surfaced as litellm.AuthenticationError, litellm.BadRequestError, etc.
      - Any other unexpected exception type.
    """
    # LiteLLM exception hierarchy mirrors OpenAI's
    if isinstance(exc, (litellm.RateLimitError,
                        litellm.ServiceUnavailableError,
                        litellm.InternalServerError,
                        litellm.Timeout)):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return True
    return False


async def _call_and_parse(
    system_prompt: str,
    user_message: str,
    ticket_id: str,
) -> TriageResult:
    """Make a single LLM call via LiteLLM and parse the response.

    Raises ValueError on truncated or unparseable output so Tenacity can retry.
    """
    resp = await acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)

    # Detect truncation *before* attempting to parse — saves a confusing
    # "no JSON found" error and gives a clear retry signal.
    if finish_reason == "length":
        raise ValueError(
            f"{ticket_id}: output truncated (finish_reason=length, "
            f"max_tokens={MAX_OUTPUT_TOKENS}). Retrying."
        )

    raw_text = choice.message.content or ""
    return _parse_response(raw_text, ticket_id)


async def _classify_one(
    ticket: Ticket,
    retriever: Callable[[Ticket], list[str]],
    semaphore: asyncio.Semaphore,
) -> TriageResult:
    """Classify a single ticket with Tenacity retries and fallback."""
    kb_chunks = retriever(ticket)
    system_prompt = _build_system_prompt(kb_chunks)
    user_message = _build_user_message(ticket)

    # Build a per-call retrier so each ticket gets its own attempt counter.
    # wait_exponential_jitter: base 1s, max 30s, with random jitter — this
    # naturally handles 429s (longer waits) without a special case.
    retrier = retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),  # 1 initial + 3 retries
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    retryable_call = retrier(_call_and_parse)

    async with semaphore:
        try:
            result = await retryable_call(
                system_prompt, user_message, ticket.ticket_id,
            )
            logger.debug(
                "%s classified: category=%s priority=%s confidence=%s",
                ticket.ticket_id, result.category, result.priority, result.confidence,
            )
            return result

        except RetryError as exc:
            logger.error(
                "%s: all attempts exhausted, returning fallback. Last error: %s",
                ticket.ticket_id, exc.last_attempt.exception(),
            )
        except Exception as exc:
            # Non-retryable error (e.g. 401 auth failure)
            logger.error(
                "%s: non-retryable error, returning fallback: %s",
                ticket.ticket_id, exc,
            )

    return TriageResult(
        ticket_id=ticket.ticket_id,
        category="unknown",
        priority="medium",
        response="Thank you for contacting Steadfast support. We've received your ticket and a team member will follow up shortly.",
        confidence=0.0,
        flags=["llm_failure"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def classify_tickets(
    tickets: list[Ticket],
    retriever: Callable[[Ticket], list[str]],
    *,
    concurrency: int = 5,
) -> list[TriageResult]:
    """Classify a batch of tickets concurrently.

    Model id is the module-level :data:`MODEL`, set at import from ``LLM_MODEL``
    (required; see startup ``ValueError`` if unset).

    Args:
        tickets: Incoming tickets to classify.
        retriever: Callable that returns KB context chunks for a ticket.
        concurrency: Max parallel API calls.

    Returns:
        List of TriageResult in the same order as input tickets.
    """
    logger.info("Using model: %s (concurrency=%d)", MODEL, concurrency)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _classify_one(ticket, retriever, semaphore)
        for ticket in tickets
    ]

    results = await asyncio.gather(*tasks)
    logger.info(
        "Classification complete: %d tickets, %d failures",
        len(results),
        sum(1 for r in results if "llm_failure" in r.flags),
    )
    return list(results)


def classify_tickets_sync(
    tickets: list[Ticket],
    retriever: Callable[[Ticket], list[str]],
    *,
    concurrency: int = 5,
) -> list[TriageResult]:
    """Synchronous wrapper around classify_tickets for use in pipeline.py."""
    return asyncio.run(classify_tickets(tickets, retriever, concurrency=concurrency))
