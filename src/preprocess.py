"""
Stage 2: Preprocess the knowledge base.

Decisions:
- BM25 (rank_bm25.BM25Okapi) over a small embedding model: the KB vocabulary is
  highly domain-specific (product names, feature names, error phrases) so exact
  keyword overlap with incoming tickets is the dominant retrieval signal.
- Document string for indexing: subject + body + resolution — all three carry signal.
- Chunk string for the LLM prompt: structured block with category, priority,
  subject, and resolution. The resolution field is the most grounding-relevant
  part per the README ("referencing specific workarounds, configuration steps").
- k = 5 default: balances context richness against prompt token budget.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi

from models import KBEntry, Ticket

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def _tokenise(text: str) -> list[str]:
    """Lowercase and split on whitespace; removes empty tokens."""
    return [t for t in _WHITESPACE.split(text.lower().strip()) if t]


def _make_document(entry: KBEntry) -> str:
    """Single string indexed by BM25 — subject + body + resolution."""
    return " ".join([entry.subject, entry.body, entry.resolution])


def _make_chunk(entry: KBEntry) -> str:
    """Formatted context block injected into the LLM prompt."""
    return (
        f"[Category: {entry.category} | Priority: {entry.priority}]\n"
        f"Subject: {entry.subject}\n"
        f"Resolution: {entry.resolution}"
    )


def preprocess_kb(entries: list[KBEntry]) -> list[KBEntry]:
    """Return entries that have usable content for retrieval.

    An entry is kept if it has a non-empty body OR a non-empty resolution.
    Entries missing both fields provide no useful context to the LLM and are
    dropped with a warning.
    """
    kept: list[KBEntry] = []
    dropped = 0
    for e in entries:
        if e.body or e.resolution:
            kept.append(e)
        else:
            logger.warning("KB entry %s dropped — empty body and resolution", e.ticket_id)
            dropped += 1

    if dropped:
        logger.info("preprocess_kb: kept %d / %d entries (%d dropped)", len(kept), len(entries), dropped)
    else:
        logger.info("preprocess_kb: all %d entries kept", len(kept))

    return kept


def build_retriever(
    processed_kb: list[KBEntry],
    k: int = 5,
) -> Callable[[Ticket], list[str]]:
    """Build a BM25 index over the KB and return a retriever callable.

    The returned function accepts a Ticket and returns a list of up to k
    formatted context strings (chunks) ranked by BM25 relevance.

    Args:
        processed_kb: KB entries from preprocess_kb.
        k: Number of top entries to return per query.

    Returns:
        retriever(ticket) -> list[str]
    """
    if not processed_kb:
        logger.warning("build_retriever: empty KB — retriever will always return []")

        def _empty_retriever(_ticket: Ticket) -> list[str]:
            return []

        return _empty_retriever

    documents = [_tokenise(_make_document(e)) for e in processed_kb]
    index = BM25Okapi(documents)
    chunks = [_make_chunk(e) for e in processed_kb]

    def retriever(ticket: Ticket) -> list[str]:
        query_tokens = _tokenise(f"{ticket.subject} {ticket.body}")
        if not query_tokens:
            return []

        scores = index.get_scores(query_tokens)
        # argsort descending; take top k (fewer if KB is smaller than k)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k]
        return [chunks[i] for i in top_k]

    return retriever
