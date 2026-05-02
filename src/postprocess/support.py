"""Shared helpers for heuristic rules (priority bumps, ticket text)."""

from __future__ import annotations

import logging

from models import PRIORITY_RANK, Ticket, TriageResult

logger = logging.getLogger(__name__)


def ticket_text(ticket: Ticket) -> str:
    return f"{ticket.subject} {ticket.body}"


def bump_priority(result: TriageResult, target: str, reason: str) -> bool:
    """Bump priority to *target* if it's currently lower.  Returns True if bumped."""
    current_rank = PRIORITY_RANK.get(result.priority, 1)
    target_rank = PRIORITY_RANK.get(target, 1)
    if target_rank > current_rank:
        old = result.priority
        result.priority = target
        result.flags.append(f"priority_bumped:{old}->{target}({reason})")
        logger.debug(
            "%s: priority bumped %s → %s (%s)",
            result.ticket_id,
            old,
            target,
            reason,
        )
        return True
    return False
