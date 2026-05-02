from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from models import Ticket

from .base import EvaluationMetric, JoinContext

_ACTIONABILITY_RE = re.compile(
    r"please\s+try|follow\s+(these\s+steps|the\b)|\bopened\s+(a\s+)?ticket\b|\blogged\b.*\bticket\b|"
    r"check\s+your\s+settings|try\s+(the\s+following|this\b)|\b(?:see|refer\s+to)\s+kb-|docs\.steadfast|"
    r"follow\s+(?:these|the)\s+steps|next\s+steps|we\s+(?:have\s+)?(?:opened|logged)",
    re.IGNORECASE,
)


class ActionabilityHintMetric(EvaluationMetric):
    def __init__(self) -> None:
        super().__init__()
        self._hints: list[float] = []

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
        act = 1.0 if _ACTIONABILITY_RE.search(resp) else 0.0
        self._hints.append(act)
        ticket_row["actionability_hint"] = int(act)

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        mean_h = sum(self._hints) / len(self._hints) if self._hints else 0.0
        return {"actionability_hint_rate": round(mean_h, 4)}
