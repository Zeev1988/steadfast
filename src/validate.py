"""
Stage 4: Validate LLM output.

Every TriageResult coming out of Stage 3 is checked for:
  1. Category is a valid enum value.
  2. Priority is a valid enum value.
  3. Response is non-empty and meets a minimum length threshold.
  4. Confidence (if present) is in [0, 1].
  5. ticket_id is non-empty.

Invalid fields are corrected in place with safe defaults and a flag is appended
so downstream stages (heuristics, evaluation, error analysis) can see what was
fixed.  Results that already carry an "llm_failure" flag are passed through
unchanged — they were already marked as fallback in Stage 3.

After validating the full batch, a ValidationReport is returned with counts of
each issue type so the pipeline can log failure rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from models import VALID_CATEGORIES, VALID_PRIORITIES, TriageResult

logger = logging.getLogger(__name__)

# Minimum response length (chars) to be considered useful.
# Anything shorter is likely a stub or parsing artefact.
MIN_RESPONSE_LENGTH = 20

# Defaults applied when a field is invalid.
FALLBACK_CATEGORY = "unknown"
FALLBACK_PRIORITY = "medium"


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Aggregated validation statistics for a batch of results."""

    total: int = 0
    valid: int = 0
    fixed: int = 0
    issues: dict[str, int] = field(default_factory=dict)

    def record_issue(self, issue_type: str) -> None:
        self.issues[issue_type] = self.issues.get(issue_type, 0) + 1

    @property
    def failure_rate(self) -> float:
        return (self.fixed / self.total) if self.total else 0.0

    def summary(self) -> dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "fixed": self.fixed,
            "failure_rate": round(self.failure_rate, 4),
            "issues": dict(self.issues),
        }


# ---------------------------------------------------------------------------
# Single-result validation
# ---------------------------------------------------------------------------

def _validate_one(result: TriageResult, report: ValidationReport) -> TriageResult:
    """Validate and fix a single TriageResult in place.

    Returns the (possibly modified) result.  Flags are appended for each
    correction so the audit trail is preserved.
    """
    report.total += 1

    # Skip results already marked as total failures — nothing to fix.
    if "llm_failure" in result.flags:
        report.record_issue("llm_failure_passthrough")
        report.fixed += 1
        return result

    was_fixed = False

    # --- Category ---
    if result.category not in VALID_CATEGORIES:
        logger.warning(
            "%s: invalid category %r → %r",
            result.ticket_id, result.category, FALLBACK_CATEGORY,
        )
        report.record_issue("invalid_category")
        result.category = FALLBACK_CATEGORY
        result.flags.append("invalid_category")
        was_fixed = True

    # --- Priority ---
    if result.priority not in VALID_PRIORITIES:
        logger.warning(
            "%s: invalid priority %r → %r",
            result.ticket_id, result.priority, FALLBACK_PRIORITY,
        )
        report.record_issue("invalid_priority")
        result.priority = FALLBACK_PRIORITY
        result.flags.append("invalid_priority")
        was_fixed = True

    # --- Response ---
    if not result.response or not result.response.strip():
        logger.warning("%s: empty response → flagged", result.ticket_id)
        report.record_issue("empty_response")
        result.response = (
            "Thank you for contacting Steadfast support. "
            "We've received your ticket and a team member will follow up shortly."
        )
        result.flags.append("empty_response")
        was_fixed = True
    elif len(result.response.strip()) < MIN_RESPONSE_LENGTH:
        logger.warning(
            "%s: response too short (%d chars) → flagged",
            result.ticket_id, len(result.response.strip()),
        )
        report.record_issue("short_response")
        result.flags.append("short_response")
        was_fixed = True

    # --- Confidence ---
    if result.confidence is not None:
        if not (0.0 <= result.confidence <= 1.0):
            logger.warning(
                "%s: confidence %s out of range → clamped",
                result.ticket_id, result.confidence,
            )
            report.record_issue("confidence_out_of_range")
            result.confidence = max(0.0, min(1.0, result.confidence))
            was_fixed = True

    # --- ticket_id ---
    if not result.ticket_id or not result.ticket_id.strip():
        logger.warning("Result with empty ticket_id → flagged")
        report.record_issue("empty_ticket_id")
        result.flags.append("empty_ticket_id")
        was_fixed = True

    if was_fixed:
        report.fixed += 1
    else:
        report.valid += 1

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_results(results: list[TriageResult]) -> tuple[list[TriageResult], ValidationReport]:
    """Validate a batch of TriageResults from Stage 3.

    Args:
        results: Raw LLM classification results.

    Returns:
        (validated_results, report) — the results list (modified in place) and
        a ValidationReport with aggregate statistics.
    """
    report = ValidationReport()

    validated = [_validate_one(r, report) for r in results]

    logger.info(
        "Validation complete: %d total, %d valid, %d fixed (%.1f%% failure rate)",
        report.total,
        report.valid,
        report.fixed,
        report.failure_rate * 100,
    )
    if report.issues:
        logger.info("Validation issues: %s", report.issues)

    return validated, report
