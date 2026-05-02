"""Tests for Stage 6 — evaluate.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluate import (
    PRIORITY_RANK,
    PriorityCostMetric,
    evaluate_run,
    kb_alignment_proxy,
    load_eval_labels,
)
from models import Ticket


@pytest.fixture(scope="session")
def knowledge_base_csv() -> Path:
    root = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.csv"
    assert root.exists(), "data/knowledge_base.csv required for kb_proxy tests"
    return root


def _write_eval_tmp(path: Path, tickets: list[dict]) -> None:
    """Tickets must satisfy loader (ticket_id, subject, body) plus labels."""
    path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


def test_load_eval_labels_roundtrip_keyword(tmp_path: Path) -> None:
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "E-1",
                "subject": "s",
                "body": "b",
                "expected_category": "integration",
                "expected_priority": "medium",
                "expected_keywords": ["alpha", "beta"],
            }
        ],
    )
    lbl = load_eval_labels(p)
    assert lbl["E-1"]["expected_category"] == "integration"
    assert lbl["E-1"]["expected_keywords"] == ["alpha", "beta"]


def test_category_accuracy_three_tickets(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    """Two of three classifications match gold category."""
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "t1",
                "subject": "a",
                "body": "a",
                "expected_category": "bug",
                "expected_priority": "low",
            },
            {
                "ticket_id": "t2",
                "subject": "b",
                "body": "b",
                "expected_category": "billing",
                "expected_priority": "low",
            },
            {
                "ticket_id": "t3",
                "subject": "c",
                "body": "c",
                "expected_category": "bug",
                "expected_priority": "high",
            },
        ],
    )
    results = [
        {
            "ticket_id": "t1",
            "category": "bug",
            "priority": "low",
            "response": "Please try reloading the billing page docs.steadfast.io",
            "flags": [],
        },
        {
            "ticket_id": "t2",
            "category": "bug",
            "priority": "low",
            "response": "See KB-100 for next steps",
            "flags": [],
        },
        {
            "ticket_id": "t3",
            "category": "bug",
            "priority": "high",
            "response": "follow these steps now",
            "flags": [],
        },
    ]
    report = evaluate_run(results, p, kb_path=knowledge_base_csv)
    assert abs(report["category_accuracy"] - 2 / 3) < 0.002


def test_priority_under_prediction_penalizes_more_than_over(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    eval_path = tmp_path / "eval.json"
    _write_eval_tmp(
        eval_path,
        [
            {
                "ticket_id": "under",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "critical",
            },
            {
                "ticket_id": "over",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            },
        ],
    )
    results = [
        {
            "ticket_id": "under",
            "category": "bug",
            "priority": "low",
            "response": "Please try restarting",
            "flags": [],
        },
        {
            "ticket_id": "over",
            "category": "bug",
            "priority": "critical",
            "response": "Please try restarting",
            "flags": [],
        },
    ]
    report = evaluate_run(results, eval_path, kb_path=knowledge_base_csv)
    undert = next(x for x in report["per_ticket"] if x["ticket_id"] == "under")[
        "priority_cost_score"
    ]
    overt = next(x for x in report["per_ticket"] if x["ticket_id"] == "over")[
        "priority_cost_score"
    ]
    assert undert < overt
    assert (
        next(x for x in report["per_ticket"] if x["ticket_id"] == "under")[
            "priority_penalty_kind"
        ]
        == "under_prediction"
    )
    assert (
        next(x for x in report["per_ticket"] if x["ticket_id"] == "over")[
            "priority_penalty_kind"
        ]
        == "over_prediction"
    )


def test_missing_label_ticket_id_reported(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "known",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            }
        ],
    )
    results = [
        {
            "ticket_id": "ghost",
            "category": "bug",
            "priority": "low",
            "response": "x",
            "flags": [],
        },
        {
            "ticket_id": "known",
            "category": "bug",
            "priority": "low",
            "response": "please try docs.steadfast.io",
            "flags": [],
        },
    ]
    report = evaluate_run(results, p, kb_path=knowledge_base_csv)
    assert "ghost" in report["missing_label_ticket_ids"]


def test_keywords_grounding_mode_mean(tmp_path: Path) -> None:
    """When any ticket lists expected_keywords, grounding uses substring coverage."""
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "kw1",
                "subject": "s",
                "body": "b",
                "expected_category": "integration",
                "expected_priority": "medium",
                "expected_keywords": ["utf-8", "missing_kw"],
            }
        ],
    )
    results = [
        {
            "ticket_id": "kw1",
            "category": "integration",
            "priority": "medium",
            "response": "Use UTF-8 encoding for uploads.",
            "flags": [],
        }
    ]
    report = evaluate_run(results, p)
    assert report["grounding"]["mode"] == "keywords"
    assert report["grounding"]["mean"] == pytest.approx(0.5, rel=1e-6)


def test_validation_flag_rate_counts_stage4_flags(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "v1",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            },
            {
                "ticket_id": "v2",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            },
        ],
    )
    results = [
        {
            "ticket_id": "v1",
            "category": "bug",
            "priority": "low",
            "response": "ok",
            "flags": ["short_response"],
        },
        {
            "ticket_id": "v2",
            "category": "bug",
            "priority": "low",
            "response": "please try again",
            "flags": [],
        },
    ]
    report = evaluate_run(results, p, kb_path=knowledge_base_csv)
    assert report["validation_flag_rate"] == pytest.approx(0.5, rel=1e-6)


def test_priority_cost_metric_under_costs_more_than_over() -> None:
    mx = PriorityCostMetric.under_weight * (
        max(PRIORITY_RANK.values()) - min(PRIORITY_RANK.values())
    )
    ru, ku = PriorityCostMetric.calc_raw_penalty_and_kind("low", "critical", mx)
    ro, ko = PriorityCostMetric.calc_raw_penalty_and_kind("critical", "low", mx)
    assert ku == "under_prediction" and ko == "over_prediction"
    assert ru is not None and ro is not None
    assert ru > ro


def test_invalid_predicted_priority_is_max_penalty(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "bad-pri",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "medium",
            }
        ],
    )
    results = [
        {
            "ticket_id": "bad-pri",
            "category": "bug",
            "priority": "urgent",
            "response": "please try again",
            "flags": [],
        }
    ]
    report = evaluate_run(results, p, kb_path=knowledge_base_csv)
    row = report["per_ticket"][0]
    assert row["priority_penalty_kind"] == "invalid_predicted_priority"
    assert row["priority_cost_score"] == 0.0


def test_kb_alignment_is_zero_when_only_stopwords_overlap() -> None:
    t = Ticket(ticket_id="x", subject="hello", body="world")

    def fake_retriever(_: Ticket) -> list[str]:
        return ["the a an and or"]

    assert kb_alignment_proxy(t, "please the a an", fake_retriever) == 0.0


def test_high_risk_category_mismatch_indexed(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "hr1",
                "subject": "s",
                "body": "b",
                "expected_category": "billing",
                "expected_priority": "low",
            },
            {
                "ticket_id": "ok1",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            },
        ],
    )
    results = [
        {
            "ticket_id": "hr1",
            "category": "bug",
            "priority": "low",
            "response": "please try x",
            "flags": [],
            "processing_seconds": 1.0,
        },
        {
            "ticket_id": "ok1",
            "category": "bug",
            "priority": "low",
            "response": "please try y",
            "flags": [],
            "processing_seconds": 3.0,
        },
    ]
    report = evaluate_run(results, p, kb_path=knowledge_base_csv)
    assert len(report["high_risk_category_mismatches"]) == 1
    assert report["high_risk_category_mismatches"][0]["ticket_id"] == "hr1"
    assert report["high_risk_category_mismatches"][0]["expected"] == "billing"
    assert report["grounding"]["overlap_uses_filtered_tokens"] is True


def test_latency_and_avg_response_joined(
    tmp_path: Path, knowledge_base_csv: Path
) -> None:
    p = tmp_path / "eval.json"
    _write_eval_tmp(
        p,
        [
            {
                "ticket_id": "t1",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            },
            {
                "ticket_id": "t2",
                "subject": "s",
                "body": "b",
                "expected_category": "bug",
                "expected_priority": "low",
            },
        ],
    )
    results = [
        {
            "ticket_id": "t1",
            "category": "bug",
            "priority": "low",
            "response": "abc",
            "flags": [],
            "processing_seconds": 8.5,
        },
        {
            "ticket_id": "t2",
            "category": "bug",
            "priority": "low",
            "response": "defghi",
            "flags": [],
            "processing_seconds": 11.5,
        },
    ]
    report = evaluate_run(results, p, kb_path=knowledge_base_csv)
    assert report["avg_response_char_count_joined"] == pytest.approx(4.5, rel=1e-6)
    assert report["latency_seconds"] is not None
    assert report["latency_seconds"]["mean"] == pytest.approx(10.0, rel=1e-6)
    assert report["latency_seconds"]["p95"] == pytest.approx(11.5, rel=1e-6)
    assert report["latency_seconds"]["n"] == 2


def test_kb_alignment_proxy_mock_retriever() -> None:
    t = Ticket(ticket_id="x", subject="hello world", body="query text")

    def fake_retriever(_: Ticket) -> list[str]:
        return ["needle alpha beta gamma"]

    s = kb_alignment_proxy(t, "needle query", fake_retriever)
    assert s == pytest.approx(0.25, rel=1e-6)
