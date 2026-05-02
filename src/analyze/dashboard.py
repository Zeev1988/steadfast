"""Build static charts from ``eval_metrics.json`` (Stage 6 aggregate output)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from models import Category

logger = logging.getLogger(__name__)


def load_eval_metrics(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _figure_summary(metrics: dict) -> Figure:
    keys = (
        ("Category acc.", metrics.get("category_accuracy")),
        ("Priority exact", metrics.get("priority_exact_accuracy")),
        ("Priority cost μ", metrics.get("priority_cost_score_mean")),
        ("Grounding μ", metrics.get("grounding", {}).get("mean")),
        ("Actionability rate", metrics.get("actionability_hint_rate")),
        ("Validation flags", metrics.get("validation_flag_rate")),
    )
    labels: list[str] = []
    values: list[float] = []
    for label, raw in keys:
        if raw is None:
            continue
        labels.append(label)
        # Normalize to 0-100 bar scale (proportions * 100; cost mean stays 0-1).
        values.append(float(raw) * 100.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2c5282")
    ax.set_yticks(y, labels)
    ax.set_xlabel("0-100 scale (each metric times 100 when stored as 0-1)")
    ax.set_title("Evaluation summary")

    lat = metrics.get("latency_seconds") or {}
    if lat:
        note = (
            f"n={lat.get('n', '-')}  mean={lat.get('mean', '-')}s"
            f"  p95={lat.get('p95', '-')}s  max={lat.get('max', '-')}s"
        )
        ax.text(
            0.02,
            -0.18,
            note,
            transform=ax.transAxes,
            fontsize=9,
            color="#333",
        )

    fig.tight_layout()
    return fig


def _figure_confusion(metrics: dict) -> Figure | None:
    raw = metrics.get("category_confusion") or {}
    if not raw:
        return None

    order = [c.value for c in Category]
    idx = {c: i for i, c in enumerate(order)}
    n = len(order)
    mat = np.zeros((n, n))
    for key, count in raw.items():
        if "|" not in key:
            continue
        gold, pred = key.split("|", 1)
        if gold in idx and pred in idx:
            mat[idx[gold], idx[pred]] = int(count)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(mat, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, label="count")
    ax.set_xticks(np.arange(n), order, rotation=45, ha="right")
    ax.set_yticks(np.arange(n), order)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold (expected)")
    ax.set_title("Category confusion matrix (gold -> predicted)")
    fig.tight_layout()
    return fig


def _figure_priority_kinds(metrics: dict) -> Figure:
    kinds: Counter[str] = Counter()
    for row in metrics.get("per_ticket") or []:
        k = row.get("priority_penalty_kind")
        if k:
            kinds[str(k)] += 1

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels_ls = sorted(kinds.keys(), key=lambda x: (-kinds[x], x))
    counts = [kinds[lbl] for lbl in labels_ls]
    ax.bar(range(len(labels_ls)), counts, color="#285e61")
    ax.set_xticks(range(len(labels_ls)), labels_ls, rotation=35, ha="right")
    ax.set_ylabel("Tickets")
    ax.set_title("Priority penalty kind (per ticket)")
    fig.tight_layout()
    return fig


def _figure_distributions(metrics: dict) -> Figure:
    rows = metrics.get("per_ticket") or []
    overlap = [
        float(r["kb_alignment_proxy"])
        for r in rows
        if r.get("kb_alignment_proxy") is not None
    ]
    pcs = [
        float(r["priority_cost_score"])
        for r in rows
        if r.get("priority_cost_score") is not None
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    if overlap:
        ax1.hist(overlap, bins=min(12, max(4, len(overlap) // 3)), color="#744210")
        ax1.set_title("KB alignment proxy")
        ax1.set_xlabel("overlap")
        ax1.set_ylabel("count")
    else:
        ax1.set_visible(False)
    if pcs:
        ax2.hist(pcs, bins=min(12, max(4, len(pcs) // 3)), color="#553c9a")
        ax2.set_title("Priority cost score")
        ax2.set_xlabel("score")
        ax2.set_ylabel("count")
    else:
        ax2.set_visible(False)
    fig.suptitle("Per-ticket distributions")
    fig.tight_layout()
    return fig


def render_eval_dashboard(
    metrics_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "eval_",
) -> list[Path]:
    """Load metrics JSON and write PNG figures under ``output_dir``.

    Returns paths of written files.
    """
    path = Path(metrics_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_eval_metrics(path)
    written: list[Path] = []

    def save(name: str, fig: Figure) -> None:
        dest = out / f"{prefix}{name}.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(dest)
        logger.info("Wrote %s", dest)

    save("summary", _figure_summary(metrics))

    fig_c = _figure_confusion(metrics)
    if fig_c is not None:
        save("category_confusion", fig_c)

    save("priority_kinds", _figure_priority_kinds(metrics))
    save("distributions", _figure_distributions(metrics))

    return written
