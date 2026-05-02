from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .base import EvaluationMetric


def _latency_summary(seconds: Sequence[float]) -> dict[str, Any]:
    srt = sorted(seconds)
    n = len(srt)
    idx = min(max(math.ceil(0.95 * n) - 1, 0), n - 1)
    return {
        "n": n,
        "mean": round(sum(srt) / n, 4),
        "p95": round(srt[idx], 4),
        "max": round(srt[-1], 4),
    }


class LatencyMetric(EvaluationMetric):
    def __init__(self) -> None:
        super().__init__()
        self._seconds: list[float] = []

    def ingest_any_row(self, pred: dict[str, Any]) -> None:
        vlat = pred.get("processing_seconds")
        if isinstance(vlat, (int, float)) and not isinstance(vlat, bool) and vlat >= 0:
            self._seconds.append(float(vlat))

    def aggregate(self, *, n_joined: int, n_results: int) -> dict[str, Any]:
        return {
            "latency_seconds": (
                _latency_summary(self._seconds) if self._seconds else None
            ),
        }
