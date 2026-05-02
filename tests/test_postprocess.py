"""Tests for Stage 5 — postprocess.py."""

from __future__ import annotations

import pytest

from models import Ticket, TriageResult
from postprocess import CONFIDENCE_REVIEW_THRESHOLD, postprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD_RESPONSE = (
    "We've identified the issue and our team is investigating. "
    "Please check Settings > Integrations for the latest status."
)


def _make_ticket(
    ticket_id: str = "T-001",
    subject: str = "Test subject",
    body: str = "Test body",
    plan: str = "Growth",
    customer_name: str = "Acme Corp",
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        customer_name=customer_name,
        plan=plan,
        subject=subject,
        body=body,
    )


def _make_result(
    ticket_id: str = "T-001",
    category: str = "bug",
    priority: str = "medium",
    confidence: float = 0.85,
    flags: list[str] | None = None,
) -> TriageResult:
    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        response=GOOD_RESPONSE,
        confidence=confidence,
        flags=flags or [],
    )


# ---------------------------------------------------------------------------
# Large user impact → bump low to medium (V2: capped at medium)
# ---------------------------------------------------------------------------


class TestHighImpact:
    def test_100_users_bumps_low_to_medium(self) -> None:
        ticket = _make_ticket(
            subject="Dashboard slow",
            body="Affecting our 100-person team. Pages take forever.",
        )
        result = _make_result(category="performance", priority="low")
        postprocess([result], [ticket])
        assert result.priority == "medium"

    def test_entire_team_bumps_low_to_medium(self) -> None:
        ticket = _make_ticket(
            subject="Feature broken",
            body="This is blocking our entire team from working.",
        )
        result = _make_result(category="bug", priority="low")
        postprocess([result], [ticket])
        assert result.priority == "medium"

    def test_already_medium_unchanged(self) -> None:
        """V2: medium stays medium — rule only bumps low→medium."""
        ticket = _make_ticket(
            subject="Dashboard slow",
            body="Affecting our 100-person team. Pages take forever.",
        )
        result = _make_result(category="performance", priority="medium")
        postprocess([result], [ticket])
        assert result.priority == "medium"
        assert not any("priority_bumped" in f for f in result.flags)

    def test_small_impact_unchanged(self) -> None:
        ticket = _make_ticket(
            subject="Minor issue",
            body="I noticed a small display glitch on my profile page.",
        )
        result = _make_result(category="bug", priority="low")
        postprocess([result], [ticket])
        assert result.priority == "low"


# ---------------------------------------------------------------------------
# Low confidence → escalate flag
# ---------------------------------------------------------------------------


class TestLowConfidenceFlag:
    def test_low_confidence_flagged(self) -> None:
        ticket = _make_ticket()
        result = _make_result(confidence=0.4)
        postprocess([result], [ticket])
        assert "escalate_to_human" in result.flags

    def test_high_confidence_not_flagged(self) -> None:
        ticket = _make_ticket()
        result = _make_result(confidence=0.9)
        postprocess([result], [ticket])
        assert "escalate_to_human" not in result.flags

    def test_at_threshold_not_flagged(self) -> None:
        ticket = _make_ticket()
        result = _make_result(confidence=CONFIDENCE_REVIEW_THRESHOLD)
        postprocess([result], [ticket])
        assert "escalate_to_human" not in result.flags


# ---------------------------------------------------------------------------
# Multi-issue tickets
# ---------------------------------------------------------------------------


class TestMultiIssueFlag:
    def test_multiple_issues_flagged(self) -> None:
        ticket = _make_ticket(
            subject="Multiple issues with our workspace",
            body="We have several problems: Jira sync, admin access, and timeline bugs.",
        )
        result = _make_result()
        postprocess([result], [ticket])
        assert "ambiguous_category" in result.flags

    def test_single_issue_not_flagged(self) -> None:
        ticket = _make_ticket(
            subject="Dashboard slow",
            body="Our dashboard takes 12 seconds to load.",
        )
        result = _make_result()
        postprocess([result], [ticket])
        assert "ambiguous_category" not in result.flags


# ---------------------------------------------------------------------------
# llm_failure passthrough
# ---------------------------------------------------------------------------


class TestLLMFailurePassthrough:
    def test_llm_failure_skipped(self) -> None:
        ticket = _make_ticket(
            subject="Minor issue",
            body="Small question about settings.",
        )
        result = _make_result(category="security", flags=["llm_failure"])
        postprocess([result], [ticket])
        # Should NOT have been corrected — llm_failure results are skipped
        assert result.category == "security"
        assert result.flags == ["llm_failure"]


# ---------------------------------------------------------------------------
# Batch / integration tests
# ---------------------------------------------------------------------------


class TestBatchBehavior:
    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            postprocess([_make_result()], [_make_ticket(), _make_ticket()])

    def test_empty_batch(self) -> None:
        results = postprocess([], [])
        assert results == []

    def test_multiple_rules_can_fire(self) -> None:
        """Multi-issue body + blocking team → high_impact bump + ambiguous_category."""
        ticket = _make_ticket(
            subject="Multiple workspace issues",
            body="We have several problems: Jira sync broken. This is blocking our entire team.",
        )
        result = _make_result(category="bug", priority="low")
        postprocess([result], [ticket])
        assert result.priority == "medium"
        assert any("priority_bumped" in f for f in result.flags)
        assert "ambiguous_category" in result.flags
