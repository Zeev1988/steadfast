"""Rule: API / rate-limit tickets miscategorised as performance → integration."""

from __future__ import annotations

import logging
import re

from models import Ticket, TriageResult

from ..support import ticket_text

logger = logging.getLogger(__name__)

_PATTERN = re.compile(
    r"(429|rate.?limit|api.?endpoint|/v\d+/|api.?quota)", re.IGNORECASE
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if result.category == "performance" and _PATTERN.search(ticket_text(ticket)):
        result.category = "integration"
        result.flags.append("heuristic:api_rate_limit→integration")
        logger.debug("%s: API rate-limit pattern → integration", result.ticket_id)
