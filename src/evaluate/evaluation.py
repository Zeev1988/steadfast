"""Run evaluation passes over pipeline results."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from loader import load_knowledge_base, load_tickets
from models import Ticket
from preprocess import build_retriever, preprocess_kb

from .labels import any_expected_keywords, load_eval_labels
from .metrics import (
    ActionabilityHintMetric,
    CategoryAgreementMetric,
    EvaluationMetric,
    GroundingMetric,
    JoinContext,
    LatencyMetric,
    PriorityCostMetric,
    ResponseLengthJoinedMetric,
    ValidationFlagMetric,
)

logger = logging.getLogger(__name__)


def evaluate_run(
    results: list[dict[str, Any]],
    eval_labels_path: str | Path,
    *,
    kb_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate metrics comparing pipeline outputs to labeled eval tickets."""
    labels_by_id = load_eval_labels(eval_labels_path)

    keywords_mode_global = any_expected_keywords(labels_by_id)
    retriever: Callable[[Ticket], list[str]] | None = None
    if not keywords_mode_global:
        kb_p = (
            kb_path
            if kb_path is not None
            else Path(eval_labels_path).parent / "knowledge_base.csv"
        )
        kb_p = Path(kb_p)
        if kb_p.exists():
            processed_kb = preprocess_kb(load_knowledge_base(kb_p))
            retriever = build_retriever(processed_kb)
        else:
            logger.warning(
                "KB missing at %s — kb_alignment_proxy will be omitted",
                kb_p,
            )

    ticket_list = load_tickets(eval_labels_path)
    tickets_by_id = {t.ticket_id: t for t in ticket_list}

    join_ctx = JoinContext(keywords_mode_global=keywords_mode_global)

    latency_m = LatencyMetric()
    category_m = CategoryAgreementMetric()
    priority_m = PriorityCostMetric()
    validation_flag_m = ValidationFlagMetric()
    actionability_m = ActionabilityHintMetric()
    response_len_m = ResponseLengthJoinedMetric()
    grounding_m = GroundingMetric(keywords_mode_global=keywords_mode_global)

    metrics_scan: Iterable[EvaluationMetric] = (latency_m,)
    metrics_joined: tuple[EvaluationMetric, ...] = (
        category_m,
        priority_m,
        validation_flag_m,
        actionability_m,
        response_len_m,
        grounding_m,
    )

    for row in results:
        for m in metrics_scan:
            m.ingest_any_row(row)

    result_ids = {
        str(r.get("ticket_id", "")).strip() for r in results if r.get("ticket_id")
    }
    orphaned_labels = sorted(set(labels_by_id) - result_ids)
    if orphaned_labels:
        logger.warning(
            "%d labeled ticket IDs not present in results (showing first 10): %s",
            len(orphaned_labels),
            orphaned_labels[:10],
        )

    per_ticket: list[dict[str, Any]] = []
    missing_labels: list[str] = []

    for row in results:
        tid = str(row.get("ticket_id", "")).strip()
        if not tid:
            continue

        gold = labels_by_id.get(tid)
        if gold is None:
            missing_labels.append(tid)
            continue

        ticket_row: dict[str, Any] = {"ticket_id": tid}
        tk_obj = tickets_by_id.get(tid)

        for m in metrics_joined:
            m.ingest_joined(
                row,
                gold,
                ticket=tk_obj,
                retriever=retriever,
                ctx=join_ctx,
                ticket_row=ticket_row,
            )
        per_ticket.append(ticket_row)

    matched_count = sum(
        1
        for r in results
        if (rid := str(r.get("ticket_id", "")).strip()) and rid in labels_by_id
    )
    n_results = len(results)

    report: dict[str, Any] = {
        "n_results": n_results,
        "n_joined": matched_count,
        "missing_label_ticket_ids": missing_labels,
        "orphaned_label_ticket_ids": orphaned_labels,
        "per_ticket": per_ticket,
    }

    for m in (*metrics_scan, *metrics_joined):
        report.update(m.aggregate(n_joined=matched_count, n_results=n_results))

    return report
