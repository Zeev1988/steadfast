"""
Stage 7: Error analysis.

Reads pipeline results alongside gold labels and categorises every mismatch
by root cause.  Outputs a structured report (``error_analysis.json``) that
feeds back into Stage 8 (iteration).

Root-cause taxonomy
-------------------
Each error is assigned one of the following root causes:

- ``category_confusion:<expected>→<predicted>``
    The LLM picked the wrong category.  Sub-typed by the specific pair so
    recurring confusions (e.g. integration↔performance) surface clearly.
- ``priority_under_prediction``
    Predicted priority is lower than expected (miss → potential SLA breach).
- ``priority_over_prediction``
    Predicted priority is higher than expected (less harmful but noisy).
- ``grounding_miss``
    Response failed to mention any of the expected keywords, suggesting the
    KB retrieval or prompt grounding was insufficient.
- ``partial_grounding``
    Response mentioned some but not all expected keywords.
- ``llm_failure``
    LLM call failed entirely and a fallback result was used.

Design decisions
----------------
- The analysis is intentionally simple — structured enough to be machine-readable
  but readable enough to paste into the write-up directly.
- Per-ticket detail is kept so a human can drill into individual failures.
- Aggregate counts and the confusion matrix make it easy to spot systematic
  patterns (e.g. "integration is confused with performance 4 times").
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate.labels import load_eval_labels
from models import PRIORITY_RANK

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root-cause classification helpers
# ---------------------------------------------------------------------------


def _classify_priority_error(predicted: str, expected: str) -> str | None:
    """Return a root-cause tag for a priority mismatch, or None if exact."""
    p = PRIORITY_RANK.get(predicted.lower())
    e = PRIORITY_RANK.get(expected.lower())
    if p is None or e is None:
        return "priority_invalid_value"
    if p == e:
        return None
    return "priority_under_prediction" if p < e else "priority_over_prediction"


def _classify_grounding(
    response: str,
    expected_keywords: list[str] | None,
) -> str | None:
    """Return a root-cause tag based on keyword coverage in the response."""
    if not expected_keywords:
        return None
    resp_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower().strip() in resp_lower)
    if hits == 0:
        return "grounding_miss"
    if hits < len(expected_keywords):
        return "partial_grounding"
    return None  # full coverage


# ---------------------------------------------------------------------------
# Single-ticket analysis
# ---------------------------------------------------------------------------


def _analyze_one(
    result: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    """Produce an error record for one ticket.

    Returns a dict with ticket_id, any mismatches found, and a list of
    root_causes.  If the ticket is fully correct, root_causes is empty.
    """
    tid = str(result.get("ticket_id", ""))
    record: dict[str, Any] = {"ticket_id": tid, "root_causes": []}

    pred_cat = str(result.get("category", "")).strip().lower()
    exp_cat = (gold.get("expected_category") or "").strip().lower()

    pred_pri = str(result.get("priority", "")).strip().lower()
    exp_pri = (gold.get("expected_priority") or "").strip().lower()

    response = str(result.get("response", ""))
    flags = result.get("flags") or []

    # --- LLM failure (total fallback) ---
    if "llm_failure" in flags:
        record["root_causes"].append("llm_failure")
        record["llm_failure"] = True
        return record

    # --- Category ---
    if exp_cat and pred_cat != exp_cat:
        cause = f"category_confusion:{exp_cat}→{pred_cat}"
        record["root_causes"].append(cause)
        record["category_mismatch"] = {
            "expected": exp_cat,
            "predicted": pred_cat,
        }

    # --- Priority ---
    if exp_pri:
        pri_cause = _classify_priority_error(pred_pri, exp_pri)
        if pri_cause:
            record["root_causes"].append(pri_cause)
            record["priority_mismatch"] = {
                "expected": exp_pri,
                "predicted": pred_pri,
                "direction": pri_cause,
            }

    # --- Grounding (keyword coverage) ---
    expected_kws = gold.get("expected_keywords")
    if isinstance(expected_kws, list) and expected_kws:
        grounding_cause = _classify_grounding(response, expected_kws)
        if grounding_cause:
            record["root_causes"].append(grounding_cause)
            resp_lower = response.lower()
            record["grounding"] = {
                "expected_keywords": expected_kws,
                "matched": [
                    kw for kw in expected_kws if kw.lower().strip() in resp_lower
                ],
                "missed": [
                    kw for kw in expected_kws if kw.lower().strip() not in resp_lower
                ],
            }

    return record


# ---------------------------------------------------------------------------
# Batch analysis + aggregation
# ---------------------------------------------------------------------------


def analyze_errors(
    results: list[dict[str, Any]],
    eval_labels_path: str | Path,
) -> dict[str, Any]:
    """Run error analysis over pipeline results matched to gold labels.

    Args:
        results: Pipeline output dicts (from eval_results.json).
        eval_labels_path: Path to the labeled eval set JSON.

    Returns:
        A structured report dict ready to be serialised as error_analysis.json.
    """
    labels_by_id = load_eval_labels(eval_labels_path)

    per_ticket_errors: list[dict[str, Any]] = []
    root_cause_counter: Counter[str] = Counter()
    confusion_matrix: Counter[str] = Counter()  # "expected|predicted"

    n_correct = 0
    n_errors = 0
    n_matched = 0

    for row in results:
        tid = str(row.get("ticket_id", "")).strip()
        if not tid:
            continue
        gold = labels_by_id.get(tid)
        if gold is None:
            continue

        n_matched += 1
        record = _analyze_one(row, gold)

        if record["root_causes"]:
            n_errors += 1
            per_ticket_errors.append(record)
            for cause in record["root_causes"]:
                root_cause_counter[cause] += 1
        else:
            n_correct += 1

        # Build confusion matrix for category
        exp_cat = (gold.get("expected_category") or "").strip().lower()
        pred_cat = str(row.get("category", "")).strip().lower()
        if exp_cat:
            confusion_matrix[f"{exp_cat}|{pred_cat}"] += 1

    # --- Build readable confusion summary ---
    category_confusions: list[dict[str, Any]] = []
    for key, count in confusion_matrix.most_common():
        exp, pred = key.split("|", 1)
        if exp != pred:
            category_confusions.append(
                {"expected": exp, "predicted": pred, "count": count}
            )

    # --- Group errors by root cause for the summary ---
    root_cause_groups: dict[str, list[str]] = {}
    for record in per_ticket_errors:
        for cause in record["root_causes"]:
            root_cause_groups.setdefault(cause, []).append(record["ticket_id"])

    report: dict[str, Any] = {
        "summary": {
            "total_evaluated": n_matched,
            "total_correct": n_correct,
            "total_with_errors": n_errors,
            "error_rate": round(n_errors / n_matched, 4) if n_matched else 0.0,
        },
        "root_cause_counts": dict(root_cause_counter.most_common()),
        "root_cause_ticket_ids": root_cause_groups,
        "category_confusions": category_confusions,
        "confusion_matrix": dict(confusion_matrix.most_common()),
        "per_ticket_errors": per_ticket_errors,
    }

    logger.info(
        "Error analysis: %d/%d tickets have errors (%d root causes)",
        n_errors,
        n_matched,
        len(root_cause_counter),
    )

    return report
