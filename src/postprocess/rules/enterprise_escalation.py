"""Rule: Enterprise plan + high-severity issue → consider critical."""

from __future__ import annotations

from models import Ticket, TriageResult

from ..support import bump_priority


def apply(result: TriageResult, ticket: Ticket) -> None:
    if ticket.plan.lower() == "enterprise" and result.priority == "high":
        if result.category in ("performance", "integration", "security", "bug"):
            bump_priority(result, "critical", "enterprise_high_severity")
