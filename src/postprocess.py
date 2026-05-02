"""
Stage 5: Heuristics & post-processing rules.

Rule-based corrections applied on top of LLM output (Stage 3 → 4 → **5**).
Each rule targets a specific confusion zone identified through data analysis
of the 300-ticket KB and 40-ticket eval set.

Design decisions
----------------
- Rules are ordered from most specific (keyword overrides) to most general
  (plan-based priority bumps).  A result can be modified by multiple rules;
  every correction appends a flag so the audit trail is transparent.
- Rules only *correct* — they don't re-classify from scratch.  If the LLM
  got it right, the rules leave it alone.
- Priority can be bumped up (never down) by heuristics, matching the
  principle that a missed escalation is worse than a false escalation.
- All thresholds (keyword lists, confidence cutoffs) are constants at the
  top of the file for easy tuning during iteration (Stage 8).
"""

from __future__ import annotations

import logging
import re

from models import Ticket, TriageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority ordering (for bump-up logic)
# ---------------------------------------------------------------------------
_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_RANK_TO_PRIORITY = {v: k for k, v in _PRIORITY_RANK.items()}


def _bump_priority(result: TriageResult, target: str, reason: str) -> bool:
    """Bump priority to *target* if it's currently lower.  Returns True if bumped."""
    current_rank = _PRIORITY_RANK.get(result.priority, 1)
    target_rank = _PRIORITY_RANK.get(target, 1)
    if target_rank > current_rank:
        old = result.priority
        result.priority = target
        result.flags.append(f"priority_bumped:{old}->{target}({reason})")
        logger.debug(
            "%s: priority bumped %s → %s (%s)",
            result.ticket_id, old, target, reason,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Rule 1: API / rate-limit tickets miscategorised as performance → integration
# ---------------------------------------------------------------------------
# Pattern: "429", "rate limit", "/v1/", "/v2/", "API endpoint" + classified as
# performance → should be integration (API quota issue, not system slowness).

_API_RATE_LIMIT_RE = re.compile(
    r"(429|rate.?limit|api.?endpoint|/v\d+/|api.?quota)", re.IGNORECASE
)


def _rule_api_rate_limit_is_integration(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if result.category == "performance" and _API_RATE_LIMIT_RE.search(text):
        result.category = "integration"
        result.flags.append("heuristic:api_rate_limit→integration")
        logger.debug("%s: API rate-limit pattern → integration", result.ticket_id)


# ---------------------------------------------------------------------------
# Rule 2: SSO / SAML / OAuth issues → integration (not security or account)
# ---------------------------------------------------------------------------
# LLMs sometimes put SSO redirect loops under "security" or "account" when
# the root cause is an identity-provider integration issue.

_SSO_RE = re.compile(
    r"\b(sso|saml|oauth|okta|azure.?ad|idp|identity.?provider|redirect.?loop"
    r"|scim|provisioning)\b",
    re.IGNORECASE,
)


def _rule_sso_is_integration(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if result.category in ("security", "account") and _SSO_RE.search(text):
        result.category = "integration"
        result.flags.append("heuristic:sso→integration")
        logger.debug("%s: SSO/SAML pattern → integration", result.ticket_id)


# ---------------------------------------------------------------------------
# Rule 3: "How do I …" / setup guidance → onboarding (not feature_request)
# ---------------------------------------------------------------------------
# When users ask how to use an existing feature, it's onboarding — even if
# the phrasing sounds like they want something new.

_ONBOARDING_RE = re.compile(
    r"(how\s+do\s+i|how\s+to|where\s+(do\s+i|can\s+i)\s+find"
    r"|getting\s+started|set\s+up\s+for|just\s+signed\s+up"
    r"|new\s+to\s+steadfast|first\s+time\s+using"
    r"|help\s+(us\s+)?understand|need\s+guidance)",
    re.IGNORECASE,
)


def _rule_howto_is_onboarding(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if result.category == "feature_request" and _ONBOARDING_RE.search(text):
        result.category = "onboarding"
        result.flags.append("heuristic:howto→onboarding")
        logger.debug("%s: how-to pattern → onboarding", result.ticket_id)


# ---------------------------------------------------------------------------
# Rule 5: Explicit pricing / charge keywords → billing (not account)
# ---------------------------------------------------------------------------

_BILLING_RE = re.compile(
    r"\b(invoice|charged?\b|refund|payment|pricing|per.?seat\s+charge"
    r"|prorated|billing\s+page|expense\s+report|duplicate\s+charge"
    r"|credit\s+card)\b",
    re.IGNORECASE,
)


def _rule_money_is_billing(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if result.category == "account" and _BILLING_RE.search(text):
        result.category = "billing"
        result.flags.append("heuristic:money→billing")
        logger.debug("%s: pricing/charge pattern → billing", result.ticket_id)


# ---------------------------------------------------------------------------
# Rule 6: Data loss / security breach → bump to critical
# ---------------------------------------------------------------------------

_CRITICAL_RE = re.compile(
    r"(data\s+loss|files?\s+disappear|losing\s+(data|files|deliverables)"
    r"|security\s+breach|unauthorized\s+access|account\s+compromised"
    r"|can'?t\s+log\s*in.*(?:all|every|most|half)\s+(?:of\s+)?(?:our\s+)?users)",
    re.IGNORECASE,
)


def _rule_critical_keywords(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if _CRITICAL_RE.search(text):
        _bump_priority(result, "critical", "data_loss_or_breach")


# ---------------------------------------------------------------------------
# Rule 7: Large user impact → bump to at least high
# ---------------------------------------------------------------------------
# Phrases like "100-person team", "50 users affected", "entire team" signal
# broad impact that warrants at least high priority.

_HIGH_IMPACT_RE = re.compile(
    r"(\d{2,}\s*[\-+]?\s*(users?|people|person|team\s+members?|employees?)"
    r"|entire\s+team|whole\s+(team|company|org)"
    r"|everyone\s+(is|in\s+our)|all\s+(of\s+)?our\s+users"
    r"|blocking\s+(our|the)\s+(team|workflow|pipeline))",
    re.IGNORECASE,
)


def _rule_high_impact(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if _HIGH_IMPACT_RE.search(text):
        _bump_priority(result, "high", "large_user_impact")


# ---------------------------------------------------------------------------
# Rule 8: Enterprise plan + high-severity issue → consider critical
# ---------------------------------------------------------------------------
# Enterprise outages have outsized business impact.

def _rule_enterprise_escalation(
    result: TriageResult, ticket: Ticket,
) -> None:
    if ticket.plan.lower() == "enterprise" and result.priority == "high":
        # Only escalate for categories where outages are plausible
        if result.category in ("performance", "integration", "security", "bug"):
            _bump_priority(result, "critical", "enterprise_high_severity")


# ---------------------------------------------------------------------------
# Rule 9: Low-confidence results → flag for human review
# ---------------------------------------------------------------------------

CONFIDENCE_REVIEW_THRESHOLD = 0.6


def _rule_low_confidence_flag(
    result: TriageResult, _ticket: Ticket,
) -> None:
    if result.confidence is not None and result.confidence < CONFIDENCE_REVIEW_THRESHOLD:
        if "escalate_to_human" not in result.flags:
            result.flags.append("escalate_to_human")
            logger.debug("%s: low confidence (%.2f) → escalate", result.ticket_id, result.confidence)


# ---------------------------------------------------------------------------
# Rule 10: Multi-issue tickets → flag for human review
# ---------------------------------------------------------------------------
# Tickets that mention multiple distinct problems should be reviewed by a human
# who can split them or pick the primary issue.

_MULTI_ISSUE_RE = re.compile(
    r"(multiple\s+issues|several\s+problems|three\s+things|a\s+few\s+issues"
    r"|first\s*[,:].*second\s*[,:].*third|issue\s+\d\s*[:.]\s*.*issue\s+\d)",
    re.IGNORECASE,
)


def _rule_multi_issue_flag(
    result: TriageResult, ticket: Ticket,
) -> None:
    text = f"{ticket.subject} {ticket.body}"
    if _MULTI_ISSUE_RE.search(text):
        if "ambiguous_category" not in result.flags:
            result.flags.append("ambiguous_category")
            logger.debug("%s: multi-issue ticket → flagged", result.ticket_id)


# ---------------------------------------------------------------------------
# Ordered rule list
# ---------------------------------------------------------------------------
# Most specific category corrections first, then priority bumps, then flags.

_RULES = [
    # Category corrections (specific → general)
    _rule_api_rate_limit_is_integration,   # Rule 1
    _rule_sso_is_integration,              # Rule 2
    _rule_howto_is_onboarding,             # Rule 3
    _rule_money_is_billing,                # Rule 4
    # Priority bumps (never down, only up)
    _rule_critical_keywords,               # Rule 5
    _rule_high_impact,                     # Rule 6
    _rule_enterprise_escalation,           # Rule 7
    # Flags
    _rule_low_confidence_flag,             # Rule 8
    _rule_multi_issue_flag,                # Rule 9
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess(
    results: list[TriageResult],
    tickets: list[Ticket],
) -> list[TriageResult]:
    """Apply heuristic rules to a batch of validated results.

    Args:
        results: Validated TriageResults from Stage 4 (same order as tickets).
        tickets: Original Ticket objects (needed for text matching).

    Returns:
        The same list of TriageResult objects, modified in place.
    """
    if len(results) != len(tickets):
        raise ValueError(
            f"results ({len(results)}) and tickets ({len(tickets)}) must be same length"
        )

    corrections = 0
    for result, ticket in zip(results, tickets, strict=True):
        # Skip results that already failed completely
        if "llm_failure" in result.flags:
            continue

        flags_before = len(result.flags)
        for rule_fn in _RULES:
            rule_fn(result, ticket)

        if len(result.flags) > flags_before:
            corrections += 1

    logger.info(
        "Post-processing complete: %d/%d results modified",
        corrections, len(results),
    )
    return results
