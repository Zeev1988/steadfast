"""Tests for Stage 4 — validate.py."""

from __future__ import annotations

import pytest

from models import VALID_CATEGORIES, VALID_PRIORITIES, TriageResult
from validate import (
    FALLBACK_CATEGORY,
    FALLBACK_PRIORITY,
    MIN_RESPONSE_LENGTH,
    ValidationReport,
    validate_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD_RESPONSE = (
    "We've identified the issue with your dashboard loading times. "
    "Our team has added Redis caching for widget data which should "
    "reduce load times significantly. Please try again and let us know."
)

FALLBACK_RESPONSE_FRAGMENT = "Thank you for contacting Steadfast support"


def _make_result(
    ticket_id: str = "EVAL-001",
    category: str = "bug",
    priority: str = "high",
    response: str = GOOD_RESPONSE,
    confidence: float | None = 0.85,
    flags: list[str] | None = None,
) -> TriageResult:
    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        response=response,
        confidence=confidence,
        flags=flags or [],
    )


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_empty_report(self) -> None:
        r = ValidationReport()
        assert r.total == 0 and r.valid == 0 and r.fixed == 0
        assert r.failure_rate == 0.0
        assert r.issues == {}

    def test_record_issue_counts(self) -> None:
        r = ValidationReport()
        r.record_issue("invalid_category")
        r.record_issue("invalid_category")
        r.record_issue("empty_response")
        assert r.issues == {"invalid_category": 2, "empty_response": 1}

    def test_failure_rate(self) -> None:
        r = ValidationReport(total=10, fixed=3)
        assert r.failure_rate == 0.3

    def test_summary_shape(self) -> None:
        r = ValidationReport(total=5, valid=4, fixed=1)
        r.record_issue("invalid_priority")
        s = r.summary()
        assert set(s.keys()) == {"total", "valid", "fixed", "failure_rate", "issues"}
        assert s["failure_rate"] == 0.2


# ---------------------------------------------------------------------------
# Valid results — should pass through unchanged
# ---------------------------------------------------------------------------


class TestValidResults:
    def test_valid_result_no_flags(self) -> None:
        result = _make_result()
        validated, report = validate_results([result])
        assert validated[0].flags == []
        assert report.valid == 1 and report.fixed == 0

    def test_valid_result_preserves_all_fields(self) -> None:
        result = _make_result(
            ticket_id="T-99", category="integration", priority="critical",
            response=GOOD_RESPONSE, confidence=0.95,
        )
        validated, _ = validate_results([result])
        r = validated[0]
        assert r.ticket_id == "T-99"
        assert r.category == "integration"
        assert r.priority == "critical"
        assert r.response == GOOD_RESPONSE
        assert r.confidence == 0.95

    @pytest.mark.parametrize("cat", sorted(VALID_CATEGORIES))
    def test_all_valid_categories_accepted(self, cat: str) -> None:
        result = _make_result(category=cat)
        validated, _report = validate_results([result])
        assert validated[0].category == cat
        assert "invalid_category" not in validated[0].flags

    @pytest.mark.parametrize("pri", sorted(VALID_PRIORITIES))
    def test_all_valid_priorities_accepted(self, pri: str) -> None:
        result = _make_result(priority=pri)
        validated, _report = validate_results([result])
        assert validated[0].priority == pri
        assert "invalid_priority" not in validated[0].flags

    def test_confidence_none_is_valid(self) -> None:
        result = _make_result(confidence=None)
        validated, report = validate_results([result])
        assert validated[0].confidence is None
        assert report.valid == 1

    def test_confidence_boundary_zero(self) -> None:
        result = _make_result(confidence=0.0)
        validated, report = validate_results([result])
        assert validated[0].confidence == 0.0
        assert report.valid == 1

    def test_confidence_boundary_one(self) -> None:
        result = _make_result(confidence=1.0)
        validated, report = validate_results([result])
        assert validated[0].confidence == 1.0
        assert report.valid == 1


# ---------------------------------------------------------------------------
# Invalid category
# ---------------------------------------------------------------------------


class TestInvalidCategory:
    def test_unknown_category_replaced(self) -> None:
        result = _make_result(category="network_issue")
        validated, report = validate_results([result])
        assert validated[0].category == FALLBACK_CATEGORY
        assert "invalid_category" in validated[0].flags
        assert report.issues["invalid_category"] == 1

    def test_empty_category_replaced(self) -> None:
        result = _make_result(category="")
        validated, _ = validate_results([result])
        assert validated[0].category == FALLBACK_CATEGORY
        assert "invalid_category" in validated[0].flags

    def test_misspelled_category_replaced(self) -> None:
        result = _make_result(category="billlng")
        validated, _ = validate_results([result])
        assert validated[0].category == FALLBACK_CATEGORY

    def test_uppercase_category_not_matched(self) -> None:
        """Enum values are lowercase; uppercase should fail validation."""
        result = _make_result(category="BUG")
        validated, _ = validate_results([result])
        assert validated[0].category == FALLBACK_CATEGORY
        assert "invalid_category" in validated[0].flags


# ---------------------------------------------------------------------------
# Invalid priority
# ---------------------------------------------------------------------------


class TestInvalidPriority:
    def test_unknown_priority_replaced(self) -> None:
        result = _make_result(priority="urgent")
        validated, report = validate_results([result])
        assert validated[0].priority == FALLBACK_PRIORITY
        assert "invalid_priority" in validated[0].flags
        assert report.issues["invalid_priority"] == 1

    def test_empty_priority_replaced(self) -> None:
        result = _make_result(priority="")
        validated, _ = validate_results([result])
        assert validated[0].priority == FALLBACK_PRIORITY

    def test_uppercase_priority_not_matched(self) -> None:
        result = _make_result(priority="HIGH")
        validated, _ = validate_results([result])
        assert validated[0].priority == FALLBACK_PRIORITY
        assert "invalid_priority" in validated[0].flags


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


class TestResponseValidation:
    def test_empty_response_replaced_and_flagged(self) -> None:
        result = _make_result(response="")
        validated, report = validate_results([result])
        assert FALLBACK_RESPONSE_FRAGMENT in validated[0].response
        assert "empty_response" in validated[0].flags
        assert report.issues["empty_response"] == 1

    def test_whitespace_only_response_treated_as_empty(self) -> None:
        result = _make_result(response="   \n\t  ")
        validated, _ = validate_results([result])
        assert "empty_response" in validated[0].flags
        assert FALLBACK_RESPONSE_FRAGMENT in validated[0].response

    def test_short_response_flagged_but_kept(self) -> None:
        short = "Got it, thanks."
        assert len(short) < MIN_RESPONSE_LENGTH
        result = _make_result(response=short)
        validated, report = validate_results([result])
        assert validated[0].response == short  # not replaced
        assert "short_response" in validated[0].flags
        assert report.issues["short_response"] == 1

    def test_response_at_min_length_is_valid(self) -> None:
        exact = "x" * MIN_RESPONSE_LENGTH
        result = _make_result(response=exact)
        validated, report = validate_results([result])
        assert "short_response" not in validated[0].flags
        assert report.valid == 1


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------


class TestConfidenceValidation:
    def test_confidence_above_one_clamped(self) -> None:
        # Use model_construct to bypass Pydantic's ge/le constraint —
        # simulates a raw dict from Stage 3 parse that slipped through.
        result = TriageResult.model_construct(
            ticket_id="T-OOR-1", category="bug", priority="high",
            response=GOOD_RESPONSE, confidence=1.5, flags=[],
        )
        validated, report = validate_results([result])
        assert validated[0].confidence == 1.0
        assert report.issues["confidence_out_of_range"] == 1

    def test_confidence_below_zero_clamped(self) -> None:
        result = TriageResult.model_construct(
            ticket_id="T-OOR-2", category="bug", priority="high",
            response=GOOD_RESPONSE, confidence=-0.3, flags=[],
        )
        validated, report = validate_results([result])
        assert validated[0].confidence == 0.0
        assert report.issues["confidence_out_of_range"] == 1


# ---------------------------------------------------------------------------
# llm_failure passthrough
# ---------------------------------------------------------------------------


class TestLLMFailurePassthrough:
    def test_llm_failure_passed_through_unchanged(self) -> None:
        result = _make_result(
            category="unknown", priority="medium",
            response="Fallback.", confidence=0.0, flags=["llm_failure"],
        )
        validated, report = validate_results([result])
        assert validated[0].flags == ["llm_failure"]
        assert validated[0].category == "unknown"  # not re-validated
        assert report.issues["llm_failure_passthrough"] == 1
        assert report.fixed == 1

    def test_llm_failure_not_double_flagged(self) -> None:
        """An llm_failure result with invalid category should NOT get
        an additional invalid_category flag — we skip validation entirely."""
        result = TriageResult.model_construct(
            ticket_id="T-LLM-2", category="garbage", priority="nonsense",
            response="x", confidence=5.0, flags=["llm_failure"],
        )
        validated, _ = validate_results([result])
        assert validated[0].flags == ["llm_failure"]
        assert "invalid_category" not in validated[0].flags


# ---------------------------------------------------------------------------
# Multiple issues in a single result
# ---------------------------------------------------------------------------


class TestMultipleIssues:
    def test_all_fixable_fields_wrong(self) -> None:
        # Use model_construct to allow out-of-range confidence (2.0)
        result = TriageResult.model_construct(
            ticket_id="T-MULTI", category="oops", priority="mega",
            response="Hi.", confidence=2.0, flags=[],
        )
        validated, report = validate_results([result])
        r = validated[0]
        assert r.category == FALLBACK_CATEGORY
        assert r.priority == FALLBACK_PRIORITY
        assert r.response == "Hi."  # kept (short, not empty)
        assert r.confidence == 1.0  # clamped
        assert "invalid_category" in r.flags
        assert "invalid_priority" in r.flags
        assert "short_response" in r.flags
        # Counts as one fixed result, not three
        assert report.fixed == 1


# ---------------------------------------------------------------------------
# Batch behaviour
# ---------------------------------------------------------------------------


class TestBatchValidation:
    def test_batch_stats_correct(self) -> None:
        batch = [
            _make_result(ticket_id="B-1"),                              # valid
            _make_result(ticket_id="B-2", category="wrong"),            # 1 fix
            _make_result(ticket_id="B-3"),                              # valid
            _make_result(ticket_id="B-4", flags=["llm_failure"]),       # passthrough
        ]
        validated, report = validate_results(batch)
        assert report.total == 4
        assert report.valid == 2
        assert report.fixed == 2  # B-2 (invalid_category) + B-4 (llm_failure)
        assert len(validated) == 4

    def test_order_preserved(self) -> None:
        ids = [f"T-{i}" for i in range(10)]
        batch = [_make_result(ticket_id=tid) for tid in ids]
        validated, _ = validate_results(batch)
        assert [r.ticket_id for r in validated] == ids

    def test_empty_batch(self) -> None:
        validated, report = validate_results([])
        assert validated == []
        assert report.total == 0
        assert report.failure_rate == 0.0
