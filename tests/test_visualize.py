"""Smoke tests for eval metrics visualization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from analyze.dashboard import load_eval_metrics, render_eval_dashboard

_MINIMAL_METRICS = {
    "n_results": 2,
    "n_joined": 2,
    "per_ticket": [
        {
            "ticket_id": "EVAL-001",
            "priority_penalty_kind": "exact",
            "priority_cost_score": 1.0,
            "kb_alignment_proxy": 0.5,
        },
        {
            "ticket_id": "EVAL-002",
            "priority_penalty_kind": "over_prediction",
            "priority_cost_score": 0.8,
            "kb_alignment_proxy": 0.1,
        },
    ],
    "latency_seconds": {"n": 2, "mean": 1.0, "p95": 2.0, "max": 3.0},
    "category_accuracy": 0.5,
    "category_confusion": {"billing|billing": 1, "billing|integration": 1},
    "priority_exact_accuracy": 0.5,
    "priority_cost_score_mean": 0.9,
    "validation_flag_rate": 0.0,
    "actionability_hint_rate": 0.5,
    "grounding": {"mode": "kb_proxy", "mean": 0.3},
}


def test_load_eval_metrics_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_MINIMAL_METRICS), encoding="utf-8")
    loaded = load_eval_metrics(p)
    assert loaded["category_accuracy"] == 0.5


def test_render_eval_dashboard_writes_pngs(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_MINIMAL_METRICS), encoding="utf-8")
    out = tmp_path / "figs"
    paths = render_eval_dashboard(p, out, prefix="t_")
    assert len(paths) == 4
    for path in paths:
        assert path.suffix == ".png"
        assert path.exists()
        assert path.stat().st_size > 500
