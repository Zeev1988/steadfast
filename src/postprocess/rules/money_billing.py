"""Rule: Explicit pricing / charge keywords → billing (not account)."""

from __future__ import annotations

import logging
import re

from models import Ticket, TriageResult

from ..support import ticket_text

logger = logging.getLogger(__name__)

_PATTERN = re.compile(
    r"\b(invoice|charged?\b|refund|payment|pricing|per.?seat\s+charge"
    r"|prorated|billing\s+page|expense\s+report|duplicate\s+charge"
    r"|credit\s+card)\b",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if result.category == "account" and _PATTERN.search(ticket_text(ticket)):
        result.category = "billing"
        result.flags.append("heuristic:money→billing")
        logger.debug("%s: pricing/charge pattern → billing", result.ticket_id)
