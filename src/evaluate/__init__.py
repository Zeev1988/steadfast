"""
Stage 6 evaluation package — metrics registry and ``evaluate_run``.

Public API mirrors the former flat ``evaluate`` module so
``from evaluate import evaluate_run`` stays valid.
"""

from __future__ import annotations

from .evaluation import evaluate_run
from .labels import load_eval_labels
from .metrics import (
    ActionabilityHintMetric,
    CategoryAgreementMetric,
    EvaluationMetric,
    GroundingEvaluator,
    GroundingMetric,
    JoinContext,
    LatencyMetric,
    PriorityCostMetric,
    ResponseLengthJoinedMetric,
    ValidationFlagMetric,
    kb_alignment_proxy,
)
from .metrics.category import HIGH_RISK_CATEGORIES
from .metrics.priority import PRIORITY_RANK
from .metrics.validation_flag import STAGE_4_ISSUE_FLAGS

__all__ = (
    "HIGH_RISK_CATEGORIES",
    "PRIORITY_RANK",
    "STAGE_4_ISSUE_FLAGS",
    "ActionabilityHintMetric",
    "CategoryAgreementMetric",
    "EvaluationMetric",
    "GroundingEvaluator",
    "GroundingMetric",
    "JoinContext",
    "LatencyMetric",
    "PriorityCostMetric",
    "ResponseLengthJoinedMetric",
    "ValidationFlagMetric",
    "evaluate_run",
    "kb_alignment_proxy",
    "load_eval_labels",
)
