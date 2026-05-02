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
# Rule 1: API rate-limit → integration
# ---------------------------------------------------------------------------


class TestApiRateLimitRule:
    def test_429_performance_becomes_integration(self) -> None:
        ticket = _make_ticket(
            subject="Getting rate limited on API",
            body="We're hitting 429 errors on /v2/analytics endpoint.",
        )
        result = _make_result(category="performance")
        postprocess([result], [ticket])
        assert result.category == "integration"
        assert any("api_rate_limit" in f for f in result.flags)

    def test_rate_limit_keyword(self) -> None:
        ticket = _make_ticket(
            subject="API rate limit issues",
            body="Our API calls are being rate limited.",
        )
        result = _make_result(category="performance")
        postprocess([result], [ticket])
        assert result.category == "integration"

    def test_does_not_change_if_already_integration(self) -> None:
        ticket = _make_ticket(
            subject="API 429 errors",
            body="Rate limit on /v2/tasks endpoint.",
        )
        result = _make_result(category="integration")
        postprocess([result], [ticket])
        assert result.category == "integration"
        assert not any("api_rate_limit" in f for f in result.flags)

    def test_does_not_change_real_performance(self) -> None:
        ticket = _make_ticket(
            subject="Dashboard very slow",
            body="Pages take 12 seconds to load for our 100-person team.",
        )
        result = _make_result(category="performance")
        postprocess([result], [ticket])
        assert result.category == "performance"


# ---------------------------------------------------------------------------
# Rule 2: SSO/SAML → integration
# ---------------------------------------------------------------------------


class TestSsoRule:
    def test_sso_security_becomes_integration(self) -> None:
        ticket = _make_ticket(
            subject="SSO login issues",
            body="Users hit a redirect loop after SSO authentication with Okta.",
        )
        result = _make_result(category="security")
        postprocess([result], [ticket])
        assert result.category == "integration"
        assert any("sso" in f for f in result.flags)

    def test_saml_account_becomes_integration(self) -> None:
        ticket = _make_ticket(
            subject="SAML configuration",
            body="Need help configuring SAML with our identity provider.",
        )
        result = _make_result(category="account")
        postprocess([result], [ticket])
        assert result.category == "integration"

    def test_scim_provisioning(self) -> None:
        ticket = _make_ticket(
            subject="SCIM user provisioning",
            body="Setting up SCIM provisioning with Azure AD.",
        )
        result = _make_result(category="account")
        postprocess([result], [ticket])
        assert result.category == "integration"

    def test_real_security_not_changed(self) -> None:
        ticket = _make_ticket(
            subject="Unauthorized access to my account",
            body="Someone logged in from an unknown IP address.",
        )
        result = _make_result(category="security")
        postprocess([result], [ticket])
        assert result.category == "security"


# ---------------------------------------------------------------------------
# Rule 3: How-to → onboarding
# ---------------------------------------------------------------------------


class TestHowtoRule:
    def test_how_do_i_feature_request_becomes_onboarding(self) -> None:
        ticket = _make_ticket(
            subject="How do I set up workflows?",
            body="How do I configure automation workflows in Steadfast?",
        )
        result = _make_result(category="feature_request")
        postprocess([result], [ticket])
        assert result.category == "onboarding"
        assert any("howto" in f for f in result.flags)

    def test_getting_started(self) -> None:
        ticket = _make_ticket(
            subject="Getting started with Steadfast",
            body="We just signed up and need guidance on setting up our workspace.",
        )
        result = _make_result(category="feature_request")
        postprocess([result], [ticket])
        assert result.category == "onboarding"

    def test_real_feature_request_unchanged(self) -> None:
        ticket = _make_ticket(
            subject="Request: calendar view",
            body="We would love a calendar view for our project tasks.",
        )
        result = _make_result(category="feature_request")
        postprocess([result], [ticket])
        assert result.category == "feature_request"


# ---------------------------------------------------------------------------
# Rule 5: Money keywords → billing
# ---------------------------------------------------------------------------


class TestBillingRule:
    def test_charge_account_becomes_billing(self) -> None:
        ticket = _make_ticket(
            subject="Unexpected charge",
            body="Why am I being charged for inactive users?",
        )
        result = _make_result(category="account")
        postprocess([result], [ticket])
        assert result.category == "billing"
        assert any("money" in f for f in result.flags)

    def test_invoice_keyword(self) -> None:
        ticket = _make_ticket(
            subject="Invoice question",
            body="Can you send the latest invoice to our finance team?",
        )
        result = _make_result(category="account")
        postprocess([result], [ticket])
        assert result.category == "billing"

    def test_real_account_unchanged(self) -> None:
        ticket = _make_ticket(
            subject="Transfer workspace ownership",
            body="Our admin left and we need to transfer ownership.",
        )
        result = _make_result(category="account")
        postprocess([result], [ticket])
        assert result.category == "account"


# ---------------------------------------------------------------------------
# Rule 6: Data loss / breach → critical
# ---------------------------------------------------------------------------


class TestCriticalKeywords:
    def test_data_loss_bumps_to_critical(self) -> None:
        ticket = _make_ticket(
            subject="Files disappearing",
            body="Files uploaded to tasks are disappearing after 24 hours. Data loss.",
        )
        result = _make_result(category="bug", priority="high")
        postprocess([result], [ticket])
        assert result.priority == "critical"
        assert any("priority_bumped" in f for f in result.flags)

    def test_security_breach(self) -> None:
        ticket = _make_ticket(
            subject="Account compromised",
            body="Unauthorized access detected. Security breach.",
        )
        result = _make_result(category="security", priority="high")
        postprocess([result], [ticket])
        assert result.priority == "critical"

    def test_already_critical_unchanged(self) -> None:
        ticket = _make_ticket(
            subject="Data loss",
            body="Files disappearing. Losing important deliverables.",
        )
        result = _make_result(category="bug", priority="critical")
        postprocess([result], [ticket])
        assert result.priority == "critical"
        # No bump flag since already critical
        assert not any("priority_bumped" in f for f in result.flags)


# ---------------------------------------------------------------------------
# Rule 7: Large user impact → at least high
# ---------------------------------------------------------------------------


class TestHighImpact:
    def test_100_users_bumps_to_high(self) -> None:
        ticket = _make_ticket(
            subject="Dashboard slow",
            body="Affecting our 100-person team. Pages take forever.",
        )
        result = _make_result(category="performance", priority="medium")
        postprocess([result], [ticket])
        assert result.priority == "high"

    def test_entire_team(self) -> None:
        ticket = _make_ticket(
            subject="Feature broken",
            body="This is blocking our entire team from working.",
        )
        result = _make_result(category="bug", priority="low")
        postprocess([result], [ticket])
        assert result.priority == "high"

    def test_small_impact_unchanged(self) -> None:
        ticket = _make_ticket(
            subject="Minor issue",
            body="I noticed a small display glitch on my profile page.",
        )
        result = _make_result(category="bug", priority="low")
        postprocess([result], [ticket])
        assert result.priority == "low"


# ---------------------------------------------------------------------------
# Rule 8: Enterprise + high severity → critical
# ---------------------------------------------------------------------------


class TestEnterpriseEscalation:
    def test_enterprise_high_bug_becomes_critical(self) -> None:
        ticket = _make_ticket(
            plan="Enterprise", subject="App crash", body="System error."
        )
        result = _make_result(category="bug", priority="high")
        postprocess([result], [ticket])
        assert result.priority == "critical"
        assert any("enterprise" in f for f in result.flags)

    def test_enterprise_medium_not_bumped(self) -> None:
        ticket = _make_ticket(
            plan="Enterprise", subject="Minor bug", body="Small issue."
        )
        result = _make_result(category="bug", priority="medium")
        postprocess([result], [ticket])
        # Rule 8 only fires for high → critical, not medium → high
        # (unless another rule bumped it first)
        assert result.priority == "medium"

    def test_growth_high_not_bumped(self) -> None:
        ticket = _make_ticket(plan="Growth", subject="App crash", body="System error.")
        result = _make_result(category="bug", priority="high")
        postprocess([result], [ticket])
        assert result.priority == "high"

    def test_enterprise_feature_request_not_bumped(self) -> None:
        ticket = _make_ticket(
            plan="Enterprise", subject="Feature idea", body="Would be nice."
        )
        result = _make_result(category="feature_request", priority="high")
        postprocess([result], [ticket])
        # feature_request not in the escalation categories
        assert result.priority == "high"


# ---------------------------------------------------------------------------
# Rule 9: Low confidence → escalate flag
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
# Rule 10: Multi-issue tickets
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
            subject="SSO redirect loop",  # Would trigger Rule 2
            body="Users can't log in via SSO.",
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
        """An Enterprise SSO ticket classified as security/medium should get
        category corrected to integration AND priority eventually bumped."""
        ticket = _make_ticket(
            plan="Enterprise",
            subject="SSO broken",
            body="Okta SSO redirect loop. Blocking our entire team.",
        )
        result = _make_result(category="security", priority="medium")
        postprocess([result], [ticket])
        # Rule 2: security → integration
        assert result.category == "integration"
        # Rule 7: "entire team" → high
        # Rule 8: Enterprise + high + integration → critical
        assert result.priority == "critical"
