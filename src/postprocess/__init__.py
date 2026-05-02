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
- Per-rule constants live beside each rule module for tuned iteration (Stage 8).
"""

from __future__ import annotations

from .rules.low_confidence_flag import CONFIDENCE_REVIEW_THRESHOLD
from .run import postprocess

__all__ = ["CONFIDENCE_REVIEW_THRESHOLD", "postprocess"]
