"""Load structured gold labels from eval ticket JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_eval_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """Index eval JSON records by ticket_id."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Eval file must be a JSON array: {path}")

    out: dict[str, dict[str, Any]] = {}
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            logger.warning("Eval row %d skipped — not an object", i)
            continue
        tid = str(obj.get("ticket_id", "")).strip()
        if not tid:
            logger.warning("Eval row %d skipped — missing ticket_id", i)
            continue
        labels: dict[str, Any] = {}
        if "expected_category" in obj:
            labels["expected_category"] = str(obj["expected_category"]).strip().lower()
        if "expected_priority" in obj:
            labels["expected_priority"] = str(obj["expected_priority"]).strip().lower()
        kws = obj.get("expected_keywords")
        if isinstance(kws, list):
            labels["expected_keywords"] = [
                str(x).strip()
                for x in kws
                if isinstance(x, (str, int, float)) and str(x).strip()
            ]
        out[tid] = labels

    logger.info("Loaded eval labels for %d ticket IDs from %s", len(out), path)
    return out


def any_expected_keywords(labels_by_id: dict[str, dict[str, Any]]) -> bool:
    return any(labels.get("expected_keywords") for labels in labels_by_id.values())
