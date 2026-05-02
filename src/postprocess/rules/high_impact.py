"""Rule: Large user impact → bump low to medium.

V1 bumped any priority to ``high``, but evaluation showed this over-escalated
onboarding and billing tickets that merely mentioned team sizes (e.g. "50 users"
in a billing question, "25-person marketing team" in a setup request).

V2 caps the bump at low→medium.  Genuine high-impact outages ("blocking our
entire team") are typically already classified as high/critical by the LLM.
"""

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
        bump_priority(result, "medium", "large_user_impact")
