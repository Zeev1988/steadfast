"""Batch driver: apply heuristic rules in order."""

from __future__ import annotations

import logging

from models import Ticket, TriageResult

from .rules import RULES

logger = logging.getLogger(__name__)


def postprocess(
    results: list[TriageResult],
    tickets: list[Ticket],
) -> list[TriageResult]:
    """Apply heuristic rules to a batch of validated results.

    Args:
        results: Validated TriageResults from Stage 4 (same order as tickets).
        tickets: Original Ticket objects (needed for text matching).

    Returns:
        The same list of TriageResult objects, modified in place.
    """
    if len(results) != len(tickets):
        raise ValueError(
            f"results ({len(results)}) and tickets ({len(tickets)}) must be same length"
        )

    corrections = 0
    for result, ticket in zip(results, tickets, strict=True):
        if "llm_failure" in result.flags:
            continue

        flags_before = len(result.flags)
        for rule_fn in RULES:
            rule_fn(result, ticket)

        if len(result.flags) > flags_before:
            corrections += 1

    logger.info(
        "Post-processing complete: %d/%d results modified",
        corrections,
        len(results),
    )
    return results
