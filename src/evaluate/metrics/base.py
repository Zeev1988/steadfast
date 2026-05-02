"""Abstract metric and join-scoped context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from models import Ticket


@dataclass(frozen=True)
class JoinContext:
    """Per-run settings passed into joined-row metrics."""

    keywords_mode_global: bool


class EvaluationMetric(ABC):
    """Registry member: ingest rows then emit a partial report dict."""

    @abstractmethod
    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        """Return top-level report keys owned by this metric."""

    def ingest_any_row(self, pred: dict[str, Any]) -> None:
        """Optional scan of every pipeline result (latency, etc.)."""
        return None

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
        """Optional accumulator for labelled rows only."""
        return None
