from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from models import VALID_CATEGORIES, Ticket

from .base import EvaluationMetric, JoinContext

# Billing / security misroutes are surfaced explicitly in reports.
HIGH_RISK_CATEGORIES: frozenset[str] = frozenset({"billing", "security"})


class CategoryAgreementMetric(EvaluationMetric):
    def __init__(self) -> None:
        super().__init__()
        self._confusion: Counter[str] = Counter()
        self._cat_correct = 0
        self._denom_cat = 0
        self._high_risk_mismatches: list[dict[str, str]] = []

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
        tid = ticket_row["ticket_id"]
        pred_cat = str(pred.get("category", "")).strip().lower()
        exp_cat = (gold.get("expected_category") or "").strip().lower()

        if exp_cat and exp_cat in VALID_CATEGORIES:
            self._denom_cat += 1
            self._confusion[f"{exp_cat}|{pred_cat}"] += 1
            if pred_cat == exp_cat:
                self._cat_correct += 1
            else:
                ticket_row["category_mismatch"] = {
                    "expected": exp_cat,
                    "predicted": pred_cat,
                }
                if exp_cat in HIGH_RISK_CATEGORIES:
                    self._high_risk_mismatches.append(
                        {"ticket_id": tid, "expected": exp_cat, "predicted": pred_cat},
                    )

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        acc = self._cat_correct / self._denom_cat if self._denom_cat else 0.0
        return {
            "category_accuracy": round(acc, 4),
            "category_confusion": dict(self._confusion),
            "high_risk_category_mismatches": list(self._high_risk_mismatches),
        }
