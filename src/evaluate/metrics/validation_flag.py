from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models import Ticket

from .base import EvaluationMetric, JoinContext

STAGE_4_ISSUE_FLAGS: frozenset[str] = frozenset(
    {
        "invalid_category",
        "invalid_priority",
        "empty_response",
        "short_response",
        "empty_ticket_id",
    }
)


class ValidationFlagMetric(EvaluationMetric):
    """Eval-joined rows: ``validation_flag_rate`` (Stage 4 repair flags present)."""

    def __init__(self) -> None:
        super().__init__()
        self._flagged_joined = 0

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
        flags = pred.get("flags") or []
        if isinstance(flags, list) and any(f in STAGE_4_ISSUE_FLAGS for f in flags):
            self._flagged_joined += 1

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        rate = self._flagged_joined / n_joined if n_joined else 0.0
        return {"validation_flag_rate": round(rate, 4)}
