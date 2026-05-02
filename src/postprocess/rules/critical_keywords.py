"""Rule: Data loss / security breach → bump to critical."""

from __future__ import annotations

import re

from models import Ticket, TriageResult

from ..support import bump_priority, ticket_text

_PATTERN = re.compile(
    r"(data\s+loss|files?\s+disappear|losing\s+(data|files|deliverables)"
    r"|security\s+breach|unauthorized\s+access|account\s+compromised"
    r"|can'?t\s+log\s*in.*(?:all|every|most|half)\s+(?:of\s+)?(?:our\s+)?users)",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if _PATTERN.search(ticket_text(ticket)):
        bump_priority(result, "critical", "data_loss_or_breach")
