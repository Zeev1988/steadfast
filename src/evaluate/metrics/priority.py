from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models import PRIORITY_RANK, Ticket

from .base import EvaluationMetric, JoinContext


class PriorityCostMetric(EvaluationMetric):
    under_weight = 3.0
    over_weight = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._max_raw = self.under_weight * (
            max(PRIORITY_RANK.values()) - min(PRIORITY_RANK.values())
        )
        self._prio_scores: list[float] = []
        self._prio_exact = 0
        self._prio_scored_count = 0

    def raw_penalty_and_kind(
        self, predicted: str, gold: str
    ) -> tuple[float | None, str]:
        return self.calc_raw_penalty_and_kind(predicted, gold, self._max_raw)

    @staticmethod
    def calc_raw_penalty_and_kind(
        predicted: str,
        gold: str,
        max_raw: float,
        *,
        under_w: float = 3.0,
        over_w: float = 1.0,
    ) -> tuple[float | None, str]:
        g = PRIORITY_RANK.get(gold.lower())
        if g is None:
            return None, "unknown_expected_priority_rank"
        p = PRIORITY_RANK.get(predicted.lower())
        if p is None:
            return max_raw, "invalid_predicted_priority"
        under = max(0, g - p)
        over = max(0, p - g)
        raw = under_w * float(under) + over_w * float(over)
        if raw == 0.0:
            return 0.0, "exact"
        if under > 0:
            return raw, "under_prediction"
        return raw, "over_prediction"

    def normalized_cost(self, raw: float | None) -> float | None:
        if raw is None:
            return None
        return 1.0 - min(1.0, raw / self._max_raw)

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
        pred_pri = str(pred.get("priority", "")).strip().lower()
        pred_pri_ok = pred_pri in PRIORITY_RANK
        exp_pri = (gold.get("expected_priority") or "").strip().lower()
        exp_pri_ok = exp_pri in PRIORITY_RANK

        if not exp_pri_ok:
            return

        self._prio_scored_count += 1
        if pred_pri_ok and pred_pri == exp_pri:
            self._prio_exact += 1

        raw_pen, prio_kind = self.raw_penalty_and_kind(pred_pri, exp_pri)
        pcs = self.normalized_cost(raw_pen)
        if pcs is not None:
            self._prio_scores.append(pcs)
        ticket_row["priority_penalty_kind"] = prio_kind
        if prio_kind not in {"exact", "unknown_expected_priority_rank"}:
            ticket_row["priority_mismatch"] = {
                "expected": exp_pri,
                "predicted": pred_pri,
            }
        if pcs is not None:
            ticket_row["priority_cost_score"] = round(pcs, 4)

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        prio_exact_acc = (
            self._prio_exact / self._prio_scored_count
            if self._prio_scored_count
            else 0.0
        )
        mean_cost = (
            sum(self._prio_scores) / len(self._prio_scores)
            if self._prio_scores
            else None
        )
        return {
            "priority_exact_accuracy": round(prio_exact_acc, 4),
            "priority_cost_weights": {
                "under": self.under_weight,
                "over": self.over_weight,
            },
            "priority_cost_score_mean": (
                round(mean_cost, 4) if mean_cost is not None else None
            ),
        }
