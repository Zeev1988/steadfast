"""Keyword coverage, BM25-chunk overlap metric, and helpers (stopword-filtered overlap)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from models import Ticket
from preprocess import _tokenise

from .base import EvaluationMetric, JoinContext

_DEFAULT_GROUND_STOP = frozenset(
    "a an and are as at be been being but by can could did do for from had has have "
    "he her here him his how i if in into is it its just like may me might more "
    "my no not of on once only or other our ours should some such than thank thanks "
    "that the their them then there these they this those to too try up us very was "
    "we were what when where which while who whom why will with would you your "
    "please also any does did into over such".split()
)


class GroundingEvaluator:
    """Keyword substring coverage + KB-proxy overlap."""

    def __init__(
        self,
        stopwords: frozenset[str] | None = None,
    ) -> None:
        self._stop = stopwords if stopwords is not None else _DEFAULT_GROUND_STOP

    def meaningful_tokens(self, text: str) -> set[str]:
        return {t for t in _tokenise(text) if t not in self._stop}

    def keyword_coverage(self, response: str, keywords: Sequence[str]) -> float:
        rl = response.lower()
        hits = sum(1 for kw in keywords if kw.lower().strip() in rl)
        return hits / len(keywords) if keywords else 0.0

    def kb_overlap(
        self,
        ticket: Ticket,
        response: str,
        retriever: Callable[[Ticket], list[str]],
    ) -> float:
        chunks = retriever(ticket)
        blob = " ".join(chunks)
        ref_tokens = self.meaningful_tokens(blob)
        if not ref_tokens:
            return 0.0
        resp_tokens = self.meaningful_tokens(response)
        overlap = resp_tokens & ref_tokens
        return len(overlap) / len(ref_tokens)


_default_ground = GroundingEvaluator()


def kb_alignment_proxy(
    ticket: Ticket,
    response: str,
    retriever: Callable[[Ticket], list[str]],
) -> float:
    """Overlap of retrieval text vs response (stopword-filtered tokens)."""
    return _default_ground.kb_overlap(ticket, response, retriever)


class GroundingMetric(EvaluationMetric):
    """Keyword mode OR BM25-proxy mode."""

    def __init__(
        self,
        *,
        keywords_mode_global: bool,
        evaluator: GroundingEvaluator | None = None,
    ) -> None:
        super().__init__()
        self._keywords_global = keywords_mode_global
        self._eval = evaluator or GroundingEvaluator()
        self._keyword_vals: list[float] = []
        self._proxy_vals: list[float] = []

    def ingest_joined(
        self,
        pred: dict[str, Any],
        gold: dict[str, Any],
        *,
        ticket: Ticket | None,
        retriever: Callable[[Ticket], list[str]] | None,
        ctx: JoinContext,
        ticket_row: dict[str, Any],
    ) -> None:
        resp = str(pred.get("response", ""))
        if ctx.keywords_mode_global:
            eks = gold.get("expected_keywords") or []
            if eks:
                cov = self._eval.keyword_coverage(resp, eks)
                self._keyword_vals.append(cov)
                ticket_row["keywords_coverage"] = round(cov, 4)
        elif retriever is not None and ticket is not None:
            gx = self._eval.kb_overlap(ticket, resp, retriever)
            self._proxy_vals.append(gx)
            ticket_row["kb_alignment_proxy"] = round(gx, 4)

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        if self._keywords_global:
            if self._keyword_vals:
                gout = {
                    "mode": "keywords",
                    "mean": round(
                        sum(self._keyword_vals) / len(self._keyword_vals),
                        4,
                    ),
                }
            else:
                gout = {"mode": "keywords", "mean": None}
        elif self._proxy_vals:
            gout = {
                "mode": "kb_proxy",
                "overlap_uses_filtered_tokens": True,
                "mean": round(sum(self._proxy_vals) / len(self._proxy_vals), 4),
            }
        else:
            gout = {"mode": "none", "mean": None}
        return {"grounding": gout}
