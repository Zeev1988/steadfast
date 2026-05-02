"""Tests for Stage 2 — preprocess.py."""

from __future__ import annotations

from models import KBEntry, Ticket
from preprocess import (
    _make_chunk,
    _make_document,
    _tokenise,
    build_retriever,
    preprocess_kb,
)


def _make_kb_entry(
    ticket_id: str = "TK-1",
    subject: str = "Test subject",
    body: str = "Test body",
    category: str = "bug",
    priority: str = "low",
    resolution: str = "Test resolution",
) -> KBEntry:
    return KBEntry(
        ticket_id=ticket_id,
        subject=subject,
        body=body,
        category=category,
        priority=priority,
        resolution=resolution,
    )


def _make_ticket(subject: str = "Test", body: str = "Test body") -> Ticket:
    return Ticket(ticket_id="EVAL-1", subject=subject, body=body)


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------


def test_tokenise_lowercases() -> None:
    assert _tokenise("Hello WORLD") == ["hello", "world"]


def test_tokenise_strips_extra_whitespace() -> None:
    assert _tokenise("  foo   bar  ") == ["foo", "bar"]


def test_tokenise_empty_string() -> None:
    assert _tokenise("") == []


def test_tokenise_single_word() -> None:
    assert _tokenise("webhook") == ["webhook"]


# ---------------------------------------------------------------------------
# _make_document
# ---------------------------------------------------------------------------


def test_make_document_joins_fields() -> None:
    entry = _make_kb_entry(subject="Sub", body="Body", resolution="Fix")
    doc = _make_document(entry)
    assert "Sub" in doc
    assert "Body" in doc
    assert "Fix" in doc


# ---------------------------------------------------------------------------
# _make_chunk
# ---------------------------------------------------------------------------


def test_make_chunk_contains_category() -> None:
    entry = _make_kb_entry(category="integration")
    assert "integration" in _make_chunk(entry)


def test_make_chunk_contains_priority() -> None:
    entry = _make_kb_entry(priority="high")
    assert "high" in _make_chunk(entry)


def test_make_chunk_contains_subject() -> None:
    entry = _make_kb_entry(subject="Webhook not firing")
    assert "Webhook not firing" in _make_chunk(entry)


def test_make_chunk_contains_resolution() -> None:
    entry = _make_kb_entry(resolution="Added retry logic")
    assert "Added retry logic" in _make_chunk(entry)


# ---------------------------------------------------------------------------
# preprocess_kb
# ---------------------------------------------------------------------------


def test_preprocess_kb_keeps_entries_with_body() -> None:
    entries = [_make_kb_entry(body="Has body", resolution="")]
    result = preprocess_kb(entries)
    assert len(result) == 1


def test_preprocess_kb_keeps_entries_with_resolution() -> None:
    entries = [_make_kb_entry(body="", resolution="Has resolution")]
    result = preprocess_kb(entries)
    assert len(result) == 1


def test_preprocess_kb_drops_entries_missing_both() -> None:
    entries = [
        _make_kb_entry(ticket_id="bad", body="", resolution=""),
        _make_kb_entry(ticket_id="good", body="Has body", resolution=""),
    ]
    result = preprocess_kb(entries)
    assert len(result) == 1
    assert result[0].ticket_id == "good"


def test_preprocess_kb_empty_input() -> None:
    assert preprocess_kb([]) == []


def test_preprocess_kb_preserves_order() -> None:
    entries = [_make_kb_entry(ticket_id=str(i)) for i in range(5)]
    result = preprocess_kb(entries)
    assert [e.ticket_id for e in result] == [str(i) for i in range(5)]


# ---------------------------------------------------------------------------
# build_retriever
# ---------------------------------------------------------------------------


def test_build_retriever_returns_callable() -> None:
    retriever = build_retriever([_make_kb_entry()])
    assert callable(retriever)


def test_build_retriever_empty_kb_returns_empty() -> None:
    retriever = build_retriever([])
    chunks = retriever(_make_ticket())
    assert chunks == []


def test_retriever_returns_at_most_k_chunks() -> None:
    kb = [_make_kb_entry(ticket_id=str(i)) for i in range(10)]
    retriever = build_retriever(kb, k=3)
    chunks = retriever(_make_ticket(body="some query text"))
    assert len(chunks) <= 3


def test_retriever_returns_fewer_than_k_when_kb_is_small() -> None:
    kb = [_make_kb_entry(ticket_id="1"), _make_kb_entry(ticket_id="2")]
    retriever = build_retriever(kb, k=5)
    chunks = retriever(_make_ticket())
    assert len(chunks) == 2


def test_retriever_chunk_format() -> None:
    entry = _make_kb_entry(
        category="performance",
        priority="critical",
        subject="Dashboard slow",
        resolution="Added Redis cache",
    )
    retriever = build_retriever([entry], k=1)
    chunks = retriever(_make_ticket(body="dashboard slow loading"))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "performance" in chunk
    assert "critical" in chunk
    assert "Dashboard slow" in chunk
    assert "Added Redis cache" in chunk


def test_retriever_ranks_relevant_entry_above_unrelated() -> None:
    """A KB entry closely matching the ticket query should rank first."""
    relevant = _make_kb_entry(
        ticket_id="rel",
        subject="SSO redirect loop on login",
        body="Users stuck in redirect loop after SSO authentication",
        resolution="Fixed SAML assertion consumer URL mismatch",
        category="security",
        priority="high",
    )
    unrelated = _make_kb_entry(
        ticket_id="unrel",
        subject="Invoice not sending",
        body="Billing email not delivered to customer",
        resolution="Fixed email queue flush",
        category="billing",
        priority="low",
    )
    retriever = build_retriever([relevant, unrelated], k=2)
    chunks = retriever(_make_ticket(subject="SSO login redirect loop", body="Users cannot log in SSO keeps redirecting"))
    assert len(chunks) >= 1
    # The first chunk should be from the relevant entry
    assert "SSO" in chunks[0] or "SAML" in chunks[0] or "security" in chunks[0]


def test_retriever_with_real_kb(sample_kb_path: "Path") -> None:  # type: ignore[name-defined]  # noqa: F821
    from loader import load_knowledge_base

    kb = load_knowledge_base(sample_kb_path)
    processed = preprocess_kb(kb)
    retriever = build_retriever(processed, k=5)
    ticket = _make_ticket(
        subject="Salesforce sync stopped working",
        body="Our Salesforce integration has not synced since last Tuesday",
    )
    chunks = retriever(ticket)
    assert 1 <= len(chunks) <= 5
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)
