"""Tests for Stage 1 — loader package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from loader import inspect_kb, load_knowledge_base, load_tickets
from models import KBEntry, Ticket


def _write_kb_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "ticket_id",
        "customer_name",
        "plan",
        "subject",
        "body",
        "category",
        "priority",
        "resolution",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)


def test_load_knowledge_base_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError, match="Knowledge base not found"):
        load_knowledge_base(missing)


def test_load_knowledge_base_raises_missing_column(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("ticket_id,subject\nTK-1,Hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_knowledge_base(p)


def test_load_knowledge_base_normalises_rows(tmp_path: Path) -> None:
    p = tmp_path / "kb.csv"
    _write_kb_csv(
        p,
        [
            {
                "ticket_id": "TK-1",
                "customer_name": "Acme",
                "plan": "Starter",
                "subject": "  Sync issue  ",
                "body": "Body text",
                "category": "Integration",
                "priority": "HIGH",
                "resolution": "Fixed",
            }
        ],
    )
    rows = load_knowledge_base(p)
    assert len(rows) == 1
    assert rows[0].ticket_id == "TK-1"
    assert rows[0].category == "integration"
    assert rows[0].priority == "high"
    assert rows[0].subject == "Sync issue"
    assert isinstance(rows[0], KBEntry)


def test_load_knowledge_base_skips_empty_required_fields(tmp_path: Path) -> None:
    p = tmp_path / "kb.csv"
    _write_kb_csv(
        p,
        [
            {
                "ticket_id": "",
                "customer_name": "X",
                "plan": "Y",
                "subject": "S",
                "body": "B",
                "category": "bug",
                "priority": "low",
                "resolution": "",
            },
            {
                "ticket_id": "TK-2",
                "customer_name": "",
                "plan": "",
                "subject": "OK",
                "body": "Body",
                "category": "bug",
                "priority": "low",
                "resolution": "",
            },
        ],
    )
    rows = load_knowledge_base(p)
    assert len(rows) == 1
    assert rows[0].ticket_id == "TK-2"


def test_load_knowledge_base_fixture_file(sample_kb_path: Path) -> None:
    rows = load_knowledge_base(sample_kb_path)
    assert len(rows) >= 300
    assert all(r.ticket_id for r in rows)
    assert all(r.category for r in rows)


def test_load_tickets_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Ticket file not found"):
        load_tickets(tmp_path / "missing.json")


def test_load_tickets_raises_if_not_json_array(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_tickets(p)


def test_load_tickets_strips_fields(tmp_path: Path) -> None:
    p = tmp_path / "tickets.json"
    p.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "  EVAL-1  ",
                    "subject": "  Subj  ",
                    "body": "  Body  ",
                    "customer_name": "  Co  ",
                    "plan": "  Growth  ",
                }
            ]
        ),
        encoding="utf-8",
    )
    tickets = load_tickets(p)
    assert len(tickets) == 1
    t = tickets[0]
    assert t.ticket_id == "EVAL-1"
    assert t.subject == "Subj"
    assert t.body == "Body"
    assert t.customer_name == "Co"
    assert t.plan == "Growth"
    assert isinstance(t, Ticket)


def test_load_tickets_skips_objects_missing_required_keys(tmp_path: Path) -> None:
    p = tmp_path / "tickets.json"
    p.write_text(
        json.dumps(
            [
                {"ticket_id": "A", "subject": "S"},
                {"ticket_id": "B", "subject": "S2", "body": "B2"},
            ]
        ),
        encoding="utf-8",
    )
    tickets = load_tickets(p)
    assert len(tickets) == 1
    assert tickets[0].ticket_id == "B"


def test_load_tickets_fixture_file(sample_eval_path: Path) -> None:
    tickets = load_tickets(sample_eval_path)
    assert len(tickets) >= 40


def test_inspect_kb_empty() -> None:
    s = inspect_kb([])
    assert s["total_entries"] == 0
    assert s["avg_body_chars"] == 0
    assert s["min_body_chars"] == 0
    assert s["max_body_chars"] == 0
    assert s["avg_resolution_chars"] == 0
    assert s["min_resolution_chars"] == 0
    assert s["max_resolution_chars"] == 0
    assert s["category_distribution"] == {}
    assert s["priority_distribution"] == {}
    assert s["plan_distribution"] == {}


def test_inspect_kb_aggregates() -> None:
    entries = [
        KBEntry(
            ticket_id="1",
            plan="Starter",
            subject="s",
            body="a",
            category="bug",
            priority="low",
            resolution="ab",
        ),
        KBEntry(
            ticket_id="2",
            plan="Growth",
            subject="s2",
            body="bcd",
            category="bug",
            priority="high",
            resolution="wxyz",
        ),
    ]
    s = inspect_kb(entries)
    assert s["total_entries"] == 2
    assert s["category_distribution"] == {"bug": 2}
    assert s["priority_distribution"] == {"low": 1, "high": 1}
    assert s["plan_distribution"] == {"Starter": 1, "Growth": 1}
    assert s["avg_body_chars"] == 2
    assert s["min_body_chars"] == 1
    assert s["max_body_chars"] == 3
    assert s["avg_resolution_chars"] == 3
    assert s["min_resolution_chars"] == 2
    assert s["max_resolution_chars"] == 4
