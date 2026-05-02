"""Rule: Multi-issue tickets → flag for human review."""

from __future__ import annotations

import logging
import re

from models import Ticket, TriageResult

from ..support import ticket_text

logger = logging.getLogger(__name__)

_PATTERN = re.compile(
    r"(multiple\s+issues|several\s+problems|three\s+things|a\s+few\s+issues"
    r"|first\s*[,:].*second\s*[,:].*third|issue\s+\d\s*[:.]\s*.*issue\s+\d)",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if _PATTERN.search(ticket_text(ticket)):
        if "ambiguous_category" not in result.flags:
            result.flags.append("ambiguous_category")
            logger.debug("%s: multi-issue ticket → flagged", result.ticket_id)
