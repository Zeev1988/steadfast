"""CLI: ``python -m analyze.analyze`` (render eval metric figures)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .dashboard import render_eval_dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("analyze")


def _default_metrics_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent / "output" / "eval_metrics.json"
    )


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(
        description="Render PNG charts from eval_metrics.json",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=_default_metrics_path(),
        help="Path to eval_metrics.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "figures",
        help="Directory for PNG outputs",
    )
    p.add_argument(
        "--prefix",
        default="eval_",
        help="Filename prefix for PNGs",
    )
    args = p.parse_args(argv)

    if not args.metrics.exists():
        logger.error("Metrics file not found: %s", args.metrics)
        return 1

    paths = render_eval_dashboard(args.metrics, args.output_dir, prefix=args.prefix)
    logger.info("Done: %d figure(s)", len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
