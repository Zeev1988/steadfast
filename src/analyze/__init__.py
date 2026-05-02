"""
Analysis tools for pipeline outputs: charts from evaluation metrics.

Run::

    PYTHONPATH=src python -m analyze.analyze --metrics output/eval_metrics.json
"""

from __future__ import annotations

from .dashboard import load_eval_metrics, render_eval_dashboard

__all__ = [
    "load_eval_metrics",
    "render_eval_dashboard",
]
