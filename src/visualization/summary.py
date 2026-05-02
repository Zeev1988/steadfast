"""
KB CSV exploration: length features + label-encoded categoricals, Pearson correlation heatmap.

Run::

    PYTHONPATH=src python -m visualization.summary --csv data/knowledge_base.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loader import KB_COL_ORDER

logger = logging.getLogger(__name__)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a numeric-only frame for Pearson correlation."""
    out = pd.DataFrame(index=df.index)
    if "subject" in df.columns:
        out["subject_len"] = df["subject"].astype(str).str.len()
    if "body" in df.columns:
        out["body_len"] = df["body"].astype(str).str.len()
    if "resolution" in df.columns:
        out["resolution_len"] = df["resolution"].astype(str).str.len()
    for col in ("category", "priority", "plan"):
        if col in df.columns:
            cat = df[col].astype(str).replace("", np.nan).fillna("_missing_")
            out[f"{col}_code"] = cat.astype("category").cat.codes.astype(float)
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feats = _feature_frame(df)
    if feats.shape[1] < 2:
        raise ValueError("Not enough columns to correlate after encoding.")
    return feats.corr(numeric_only=True)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "KB CSV feature correlations (Pearson)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(corr.columns)
    mat = np.nan_to_num(corr.to_numpy(), nan=0.0)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, label="r")
    ax.set_xticks(np.arange(n), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n), labels)
    ax.set_title(title)

    raw = corr.to_numpy()
    for i in range(n):
        for j in range(n):
            val = raw[i, j]
            if np.isnan(val):
                t = "-"
                col = "#222"
            else:
                t = f"{val:.2f}"
                col = "white" if abs(val) > 0.55 else "#222"
            ax.text(j, i, t, ha="center", va="center", color=col, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote heatmap %s", out_path)


def print_summary(corr: pd.DataFrame, top_k: int = 12) -> None:
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((a, b, float(v)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("Top correlations (|r|):")
    for a, b, v in pairs[:top_k]:
        print(f"  {a!r} vs {b!r}: {v:+.3f}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(description="KB CSV correlation summary + heatmap")
    p.add_argument(
        "--csv",
        type=Path,
        default=root / "data" / "knowledge_base.csv",
        help="Knowledge base CSV path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "kb_correlation.png",
        help="PNG heatmap path",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write correlation matrix as JSON",
    )
    p.add_argument("--top", type=int, default=12, help="How many pairs to print")
    args = p.parse_args(argv)

    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        return 1

    df = pd.read_csv(
        args.csv,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    missing = set(KB_COL_ORDER) - set(df.columns)
    if missing:
        logger.error("Missing columns: %s", sorted(missing))
        return 1

    corr = correlation_matrix(df.loc[:, list(KB_COL_ORDER)])
    print_summary(corr, top_k=args.top)
    plot_correlation_heatmap(corr, args.output)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            corr.round(4).to_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote matrix JSON %s", args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
