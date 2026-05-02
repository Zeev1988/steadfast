from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models import Ticket

from .base import EvaluationMetric, JoinContext


class ResponseLengthJoinedMetric(EvaluationMetric):
    def __init__(self) -> None:
        super().__init__()
        self._lens: list[int] = []

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
        self._lens.append(len(str(pred.get("response", "")).strip()))

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        avg = round(sum(self._lens) / len(self._lens), 2) if self._lens else None
        return {"avg_response_char_count_joined": avg}
