"""Rule: SSO / SAML / OAuth issues → integration (not security or account)."""

from __future__ import annotations

import logging
import re

from models import Ticket, TriageResult

from ..support import ticket_text

logger = logging.getLogger(__name__)

_PATTERN = re.compile(
    r"\b(sso|saml|oauth|okta|azure.?ad|idp|identity.?provider|redirect.?loop"
    r"|scim|provisioning)\b",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if result.category in ("security", "account") and _PATTERN.search(
        ticket_text(ticket)
    ):
        result.category = "integration"
        result.flags.append("heuristic:sso→integration")
        logger.debug("%s: SSO/SAML pattern → integration", result.ticket_id)
