"""
Ordered heuristic rules for Stage 5.

Most specific category corrections first, then priority bumps, then flags.
"""

from __future__ import annotations

from collections.abc import Callable

from models import Ticket, TriageResult

from . import (
    high_impact,
    low_confidence_flag,
    multi_issue_flag,
)

RuleFn = Callable[[TriageResult, Ticket], None]

RULES: list[RuleFn] = [
    high_impact.apply,
    low_confidence_flag.apply,
    multi_issue_flag.apply,
]

__all__ = ["RULES", "RuleFn"]
