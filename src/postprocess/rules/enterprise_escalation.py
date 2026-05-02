"""Rule: Enterprise plan + low priority → nudge to medium.

V1 bumped high→critical for Enterprise, but evaluation showed this was wrong
7 out of 9 times (e.g. MFA setup requests, gradual slowdowns).  Enterprise
tickets are important but "Enterprise + high" does not equal "critical".
V2 restricts the bump to low→medium only — a gentle nudge that avoids
over-escalation while still ensuring Enterprise tickets don't languish.
"""

from __future__ import annotations

from models import Ticket, TriageResult

from ..support import bump_priority


def apply(result: TriageResult, ticket: Ticket) -> None:
    if ticket.plan.lower() == "enterprise" and result.priority == "low":
        bump_priority(result, "medium", "enterprise_low_bump")
