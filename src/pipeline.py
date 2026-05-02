"""
Steadfast Support Ticket Triage Pipeline

Usage:
  python src/pipeline.py                          # run on default eval set
  python src/pipeline.py --input FILE             # run on a custom ticket JSON
  python src/pipeline.py --eval                   # pipeline + evaluation + error analysis
  python src/pipeline.py --eval --limit N         # same, but process only first N tickets
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from loader import inspect_kb, load_knowledge_base, load_tickets
from preprocess import build_retriever, preprocess_kb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

DEFAULT_KB_PATH = DATA_DIR / "knowledge_base.csv"
DEFAULT_EVAL_PATH = DATA_DIR / "eval_set.json"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Steadfast triage pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        metavar="FILE",
        default=str(DEFAULT_EVAL_PATH),
        help="Path to a ticket JSON file (array of ticket objects)",
    )
    p.add_argument(
        "--kb",
        metavar="FILE",
        default=str(DEFAULT_KB_PATH),
        help="Path to the knowledge base CSV",
    )
    p.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation (stage 6) and error analysis (stage 7) after the pipeline",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N tickets (useful for dev loops)",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        default=str(OUTPUT_DIR / "eval_results.json"),
        help="Where to write the pipeline results JSON",
    )
    return p


def run_pipeline(args: argparse.Namespace) -> list[dict]:
    """Execute all pipeline stages; return list of result dicts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 — Load data
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: Load data ===")
    kb_entries = load_knowledge_base(args.kb)
    tickets = load_tickets(args.input)

    kb_summary = inspect_kb(kb_entries)
    logger.info("KB summary: %s", kb_summary)
    logger.info("Tickets to process: %d", len(tickets))

    if args.limit:
        tickets = tickets[: args.limit]
        logger.info("Limited to %d tickets (--limit)", args.limit)

    # ------------------------------------------------------------------
    # Stage 2 — Preprocess
    # ------------------------------------------------------------------
    logger.info("=== Stage 2: Preprocess ===")
    processed_kb = preprocess_kb(kb_entries)
    retriever = build_retriever(processed_kb)
    logger.info("BM25 index built over %d KB entries (k=5)", len(processed_kb))

    return []


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
