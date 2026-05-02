"""Rule: Active data loss / security breach → bump to critical.

V2 tightens the regex to require active-voice indicators of *ongoing* loss
or breach.  Hypothetical questions ("what happens to our data if we cancel?")
no longer trigger the rule — those are informational, not emergencies.

The rule now also only bumps to ``high`` instead of ``critical``, since the
LLM already assigns high priority to genuine security incidents and the bump
to critical was often excessive.
"""

from __future__ import annotations

import re

from models import Ticket, TriageResult

from ..support import bump_priority, ticket_text

# Active-voice loss/breach indicators.  Excludes hypothetical phrasing
# like "what happens to", "if we cancel", "data retention policy".
_PATTERN = re.compile(
    r"(files?\s+(are\s+)?disappear(ing|ed)"
    r"|losing\s+(data|files|deliverables)"
    r"|we\s+(lost|are\s+losing)\s+(data|files)"
    r"|security\s+breach"
    r"|unauthorized\s+access\s+(detected|to\s+our)"
    r"|account\s+(has\s+been\s+)?compromised"
    r"|can'?t\s+log\s*in.*(?:all|every|most|half)\s+(?:of\s+)?(?:our\s+)?users)",
    re.IGNORECASE,
)


def apply(result: TriageResult, ticket: Ticket) -> None:
    if _PATTERN.search(ticket_text(ticket)):
        bump_priority(result, "high", "data_loss_or_breach")
