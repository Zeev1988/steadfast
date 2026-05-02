"""Tests for Stage 7 — analyze.py (error analysis)."""

from __future__ import annotations

import json
from pathlib import Path

from analyze import analyze_errors


def _write_eval(path: Path, tickets: list[dict]) -> None:
    path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers: minimal eval set builders
# ---------------------------------------------------------------------------

_EVAL_TICKETS = [
    {
        "ticket_id": "E-1",
        "subject": "Sync broken",
        "body": "HubSpot sync stopped",
        "expected_category": "integration",
        "expected_priority": "high",
    },
    {
        "ticket_id": "E-2",
        "subject": "Invoice question",
        "body": "Charge on my card",
        "expected_category": "billing",
        "expected_priority": "low",
    },
    {
        "ticket_id": "E-3",
        "subject": "Dashboard slow",
        "body": "Page takes forever",
        "expected_category": "performance",
        "expected_priority": "medium",
    },
]


def _result(
    tid: str,
    category: str = "bug",
    priority: str = "medium",
    response: str = "We are looking into this.",
    flags: list[str] | None = None,
) -> dict:
    return {
        "ticket_id": tid,
        "category": category,
        "priority": priority,
        "response": response,
        "flags": flags or [],
    }


# ---------------------------------------------------------------------------
# All correct
# ---------------------------------------------------------------------------


class TestAllCorrect:
    def test_no_errors_when_all_match(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS)
        results = [
            _result("E-1", category="integration", priority="high"),
            _result("E-2", category="billing", priority="low"),
            _result("E-3", category="performance", priority="medium"),
        ]
        report = analyze_errors(results, p)
        assert report["summary"]["total_correct"] == 3
        assert report["summary"]["total_with_errors"] == 0
        assert report["per_ticket_errors"] == []


# ---------------------------------------------------------------------------
# Category mismatches
# ---------------------------------------------------------------------------


class TestCategoryErrors:
    def test_category_mismatch_recorded(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[:1])
        results = [_result("E-1", category="performance", priority="high")]
        report = analyze_errors(results, p)
        assert report["summary"]["total_with_errors"] == 1
        err = report["per_ticket_errors"][0]
        assert "category_confusion:integration→performance" in err["root_causes"]
        assert err["category_mismatch"]["expected"] == "integration"
        assert err["category_mismatch"]["predicted"] == "performance"

    def test_confusion_matrix_populated(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS)
        results = [
            _result("E-1", category="performance", priority="high"),
            _result("E-2", category="account", priority="low"),
            _result("E-3", category="performance", priority="medium"),
        ]
        report = analyze_errors(results, p)
        assert "integration|performance" in report["confusion_matrix"]
        assert "billing|account" in report["confusion_matrix"]
        # E-3 is correct
        assert "performance|performance" in report["confusion_matrix"]

    def test_category_confusions_list_excludes_correct(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS)
        results = [
            _result("E-1", category="integration", priority="high"),
            _result("E-2", category="account", priority="low"),
            _result("E-3", category="performance", priority="medium"),
        ]
        report = analyze_errors(results, p)
        # Only billing→account should appear
        assert len(report["category_confusions"]) == 1
        assert report["category_confusions"][0]["expected"] == "billing"


# ---------------------------------------------------------------------------
# Priority errors
# ---------------------------------------------------------------------------


class TestPriorityErrors:
    def test_under_prediction(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[:1])
        results = [_result("E-1", category="integration", priority="low")]
        report = analyze_errors(results, p)
        err = report["per_ticket_errors"][0]
        assert "priority_under_prediction" in err["root_causes"]
        assert err["priority_mismatch"]["direction"] == "priority_under_prediction"

    def test_over_prediction(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[1:2])
        results = [_result("E-2", category="billing", priority="critical")]
        report = analyze_errors(results, p)
        err = report["per_ticket_errors"][0]
        assert "priority_over_prediction" in err["root_causes"]


# ---------------------------------------------------------------------------
# Grounding (keyword coverage)
# ---------------------------------------------------------------------------


class TestGrounding:
    def test_grounding_miss(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        tickets = [
            {
                "ticket_id": "KW-1",
                "subject": "s",
                "body": "b",
                "expected_category": "integration",
                "expected_priority": "high",
                "expected_keywords": ["hubspot", "field mapping"],
            }
        ]
        _write_eval(p, tickets)
        results = [
            _result(
                "KW-1",
                category="integration",
                priority="high",
                response="We are looking into this issue.",
            )
        ]
        report = analyze_errors(results, p)
        err = report["per_ticket_errors"][0]
        assert "grounding_miss" in err["root_causes"]
        assert err["grounding"]["missed"] == ["hubspot", "field mapping"]

    def test_partial_grounding(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        tickets = [
            {
                "ticket_id": "KW-2",
                "subject": "s",
                "body": "b",
                "expected_category": "integration",
                "expected_priority": "high",
                "expected_keywords": ["hubspot", "field mapping"],
            }
        ]
        _write_eval(p, tickets)
        results = [
            _result(
                "KW-2",
                category="integration",
                priority="high",
                response="The HubSpot sync issue has been escalated.",
            )
        ]
        report = analyze_errors(results, p)
        err = report["per_ticket_errors"][0]
        assert "partial_grounding" in err["root_causes"]
        assert "hubspot" in err["grounding"]["matched"]
        assert "field mapping" in err["grounding"]["missed"]

    def test_full_grounding_no_error(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        tickets = [
            {
                "ticket_id": "KW-3",
                "subject": "s",
                "body": "b",
                "expected_category": "integration",
                "expected_priority": "high",
                "expected_keywords": ["hubspot"],
            }
        ]
        _write_eval(p, tickets)
        results = [
            _result(
                "KW-3",
                category="integration",
                priority="high",
                response="The HubSpot integration has been fixed.",
            )
        ]
        report = analyze_errors(results, p)
        assert report["summary"]["total_with_errors"] == 0


# ---------------------------------------------------------------------------
# LLM failure
# ---------------------------------------------------------------------------


class TestLLMFailure:
    def test_llm_failure_recorded(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[:1])
        results = [
            _result("E-1", category="unknown", priority="medium", flags=["llm_failure"])
        ]
        report = analyze_errors(results, p)
        err = report["per_ticket_errors"][0]
        assert "llm_failure" in err["root_causes"]
        assert err.get("llm_failure") is True
        # llm_failure should be the only cause (others are not checked)
        assert len(err["root_causes"]) == 1


# ---------------------------------------------------------------------------
# Multiple root causes
# ---------------------------------------------------------------------------


class TestMultipleCauses:
    def test_category_and_priority_both_wrong(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[:1])
        results = [_result("E-1", category="bug", priority="low")]
        report = analyze_errors(results, p)
        err = report["per_ticket_errors"][0]
        causes = err["root_causes"]
        assert any("category_confusion" in c for c in causes)
        assert "priority_under_prediction" in causes


# ---------------------------------------------------------------------------
# Root-cause grouping
# ---------------------------------------------------------------------------


class TestRootCauseGroups:
    def test_ticket_ids_grouped_by_cause(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS)
        results = [
            _result("E-1", category="performance", priority="high"),
            _result("E-2", category="account", priority="low"),
            _result("E-3", category="performance", priority="medium"),
        ]
        report = analyze_errors(results, p)
        groups = report["root_cause_ticket_ids"]
        assert "E-1" in groups.get("category_confusion:integration→performance", [])
        assert "E-2" in groups.get("category_confusion:billing→account", [])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS)
        report = analyze_errors([], p)
        assert report["summary"]["total_evaluated"] == 0
        assert report["summary"]["error_rate"] == 0.0

    def test_result_without_matching_label_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[:1])
        results = [
            _result("E-1", category="integration", priority="high"),
            _result("UNKNOWN-99", category="bug", priority="low"),
        ]
        report = analyze_errors(results, p)
        assert report["summary"]["total_evaluated"] == 1
        assert report["summary"]["total_correct"] == 1

    def test_report_structure(self, tmp_path: Path) -> None:
        p = tmp_path / "eval.json"
        _write_eval(p, _EVAL_TICKETS[:1])
        results = [_result("E-1", category="integration", priority="high")]
        report = analyze_errors(results, p)
        assert set(report.keys()) == {
            "summary",
            "root_cause_counts",
            "root_cause_ticket_ids",
            "category_confusions",
            "confusion_matrix",
            "per_ticket_errors",
        }
        assert set(report["summary"].keys()) == {
            "total_evaluated",
            "total_correct",
            "total_with_errors",
            "error_rate",
        }
