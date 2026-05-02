"""
Steadfast Support Ticket Triage Pipeline

Usage:
  python src/pipeline.py                          # run on default eval set
  python src/pipeline.py --input FILE             # run on a custom ticket JSON
  python src/pipeline.py --eval                   # pipeline + evaluation + error analysis
  python src/pipeline.py --eval --limit N         # same, but process only first N tickets

Model selection is handled in the ``agent`` package via the LLM_MODEL env var in .env.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agent import classify_tickets_sync
from evaluate import evaluate_run
from loader import inspect_kb, load_knowledge_base, load_tickets
from postprocess import postprocess
from preprocess import build_retriever, preprocess_kb
from validate import validate_results

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
    logger.info("BM25 index built over %d KB entries", len(processed_kb))
    if tickets:
        sample_chunks = retriever(tickets[0])
        preview = (sample_chunks[0][:160] + "…") if sample_chunks else "(no chunks)"
        logger.info(
            "Retriever dry-run on first ticket %s: %d chunk(s); first chunk starts: %s",
            tickets[0].ticket_id,
            len(sample_chunks),
            preview,
        )

    # ------------------------------------------------------------------
    # Stage 3 — LLM Classification
    # ------------------------------------------------------------------
    logger.info("=== Stage 3: LLM Classification ===")
    results = classify_tickets_sync(tickets, retriever)
    logger.info("Classified %d tickets", len(results))

    # ------------------------------------------------------------------
    # Stage 4 — Validate output
    # ------------------------------------------------------------------
    logger.info("=== Stage 4: Validate output ===")
    results, validation_report = validate_results(results)
    logger.info("Validation report: %s", validation_report.summary())

    # ------------------------------------------------------------------
    # Stage 5 — Heuristics & post-processing
    # ------------------------------------------------------------------
    logger.info("=== Stage 5: Heuristics & post-processing ===")
    results = postprocess(results, tickets)

    # Convert to dicts for JSON serialisation
    return [r.to_dict() for r in results]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results = run_pipeline(args)

    # Write results to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Results written to %s (%d tickets)", output_path, len(results))

    if args.eval:
        # Local import avoids circular deps when tooling imports pipeline.
        metrics_path = output_path.with_name("eval_metrics.json")
        report = evaluate_run(results, Path(args.input), kb_path=args.kb)
        with metrics_path.open("w", encoding="utf-8") as mh:
            json.dump(report, mh, indent=2, ensure_ascii=False)
        logger.info(
            "Eval metrics → %s (category_acc=%s priority_exact=%s priority_cost_mean=%s)",
            metrics_path,
            report["category_accuracy"],
            report["priority_exact_accuracy"],
            report["priority_cost_score_mean"],
        )


if __name__ == "__main__":
    main()
