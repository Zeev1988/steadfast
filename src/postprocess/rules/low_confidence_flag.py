"""Rule: Low-confidence results → flag for human review."""

from __future__ import annotations

import logging

from models import Ticket, TriageResult

logger = logging.getLogger(__name__)

CONFIDENCE_REVIEW_THRESHOLD = 0.6


def apply(result: TriageResult, _ticket: Ticket) -> None:
    if (
        result.confidence is not None
        and result.confidence < CONFIDENCE_REVIEW_THRESHOLD
    ):
        if "escalate_to_human" not in result.flags:
            result.flags.append("escalate_to_human")
            logger.debug(
                "%s: low confidence (%.2f) → escalate",
                result.ticket_id,
                result.confidence,
            )
