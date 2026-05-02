"""
Ordered heuristic rules for Stage 5.

Most specific category corrections first, then priority bumps, then flags.
"""

from __future__ import annotations

from collections.abc import Callable

from models import Ticket, TriageResult

from . import (
    api_rate_limit_integration,
    critical_keywords,
    enterprise_escalation,
    high_impact,
    howto_onboarding,
    low_confidence_flag,
    money_billing,
    multi_issue_flag,
    sso_integration,
)

RuleFn = Callable[[TriageResult, Ticket], None]

RULES: list[RuleFn] = [
    api_rate_limit_integration.apply,
    sso_integration.apply,
    howto_onboarding.apply,
    money_billing.apply,
    critical_keywords.apply,
    high_impact.apply,
    enterprise_escalation.apply,
    low_confidence_flag.apply,
    multi_issue_flag.apply,
]

__all__ = ["RULES", "RuleFn"]
