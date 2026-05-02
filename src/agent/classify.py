"""Concurrent batch classification with retries and failure fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable

import litellm
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from models import Ticket, TriageResult

from .config import MODEL
from .llm import call_and_parse
from .prompts import build_system_prompt, build_user_message

logger = logging.getLogger(__name__)


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
    if isinstance(
        exc,
        (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.Timeout,
        ),
    ):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return True
    return False


async def _classify_one(
    ticket: Ticket,
    retriever: Callable[[Ticket], list[str]],
    semaphore: asyncio.Semaphore,
) -> TriageResult:
    """Classify a single ticket with Tenacity retries and fallback."""
    kb_chunks = retriever(ticket)
    system_prompt = build_system_prompt(kb_chunks)
    user_message = build_user_message(ticket)

    retrier = retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),  # 1 initial + 3 retries
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    retryable_call = retrier(call_and_parse)

    async with semaphore:
        t0 = time.perf_counter()
        try:
            result = await retryable_call(
                system_prompt,
                user_message,
                ticket.ticket_id,
            )
            elapsed = time.perf_counter() - t0
            logger.debug(
                "%s classified: category=%s priority=%s confidence=%s",
                ticket.ticket_id,
                result.category,
                result.priority,
                result.confidence,
            )
            return result.model_copy(update={"processing_seconds": elapsed})

        except RetryError as exc:
            logger.error(
                "%s: all attempts exhausted, returning fallback. Last error: %s",
                ticket.ticket_id,
                exc.last_attempt.exception(),
            )
        except Exception as exc:
            logger.error(
                "%s: non-retryable error, returning fallback: %s",
                ticket.ticket_id,
                exc,
            )

        elapsed = time.perf_counter() - t0
        return TriageResult(
            ticket_id=ticket.ticket_id,
            category="unknown",
            priority="medium",
            response="Thank you for contacting Steadfast support. We've received your ticket and a team member will follow up shortly.",
            confidence=0.0,
            flags=["llm_failure"],
            processing_seconds=elapsed,
        )


async def classify_tickets(
    tickets: list[Ticket],
    retriever: Callable[[Ticket], list[str]],
    *,
    concurrency: int = 5,
) -> list[TriageResult]:
    """Classify a batch of tickets concurrently.

    Model id is the package-level ``MODEL``, set from ``LLM_MODEL`` at import.

    Args:
        tickets: Incoming tickets to classify.
        retriever: Callable that returns KB context chunks for a ticket.
        concurrency: Max parallel API calls.

    Returns:
        List of TriageResult in the same order as input tickets.
    """
    logger.info("Using model: %s (concurrency=%d)", MODEL, concurrency)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_classify_one(ticket, retriever, semaphore) for ticket in tickets]

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
