"""Rule: "How do I …" / setup guidance → onboarding (not feature_request)."""

from __future__ import annotations

import logging
import re

from models import Ticket, TriageResult

from ..support import ticket_text

logger = logging.getLogger(__name__)

_PATTERN = re.compile(
    r"(how\s+do\s+i|how\s+to|where\s+(do\s+i|can\s+i)\s+find"
    r"|getting\s+started|set\s+up\s+for|just\s+signed\s+up"
    r"|new\s+to\s+steadfast|first\s+time\s+using"
    r"|help\s+(us\s+)?understand|need\s+guidance)",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if result.category == "feature_request" and _PATTERN.search(ticket_text(ticket)):
        result.category = "onboarding"
        result.flags.append("heuristic:howto→onboarding")
        logger.debug("%s: how-to pattern → onboarding", result.ticket_id)
