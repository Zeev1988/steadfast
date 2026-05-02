"""Rule: Large user impact → bump to at least high."""

from __future__ import annotations

import re

from models import Ticket, TriageResult

from ..support import bump_priority, ticket_text

_PATTERN = re.compile(
    r"(\d{2,}\s*[\-+]?\s*(users?|people|person|team\s+members?|employees?)"
    r"|entire\s+team|whole\s+(team|company|org)"
    r"|everyone\s+(is|in\s+our)|all\s+(of\s+)?our\s+users"
    r"|blocking\s+(our|the)\s+(team|workflow|pipeline))",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if _PATTERN.search(ticket_text(ticket)):
        bump_priority(result, "high", "large_user_impact")
