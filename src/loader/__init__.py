"""
Stage 1: Load data.

Reads the knowledge base CSV and the eval/custom ticket JSON,
returning normalised Ticket and KBEntry objects.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from models import KBEntry, Ticket

logger = logging.getLogger(__name__)

# Keys expected in each ticket JSON object (eval set or custom input)
_TICKET_REQUIRED_KEYS = {"ticket_id", "subject", "body"}

# Knowledge base CSV columns (order matches ``tests.test_loader`` fixtures).
KB_COL_ORDER: tuple[str, ...] = (
    "ticket_id",
    "customer_name",
    "plan",
    "subject",
    "body",
    "category",
    "priority",
    "resolution",
)
_KB_REQUIRED_COLS: frozenset[str] = frozenset(KB_COL_ORDER)


def _scalar_to_str(cell: object) -> str:
    """Normalise a parsed CSV cell to a trimmed string."""
    if cell is None:
        return ""
    try:
        if pd.isna(cell):
            return ""
    except TypeError:
        pass
    return str(cell).strip()


def load_knowledge_base(path: str | Path) -> list[KBEntry]:
    """Parse the knowledge base CSV into a list of KBEntry objects.

    Rows with missing required columns are skipped with a warning.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    entries: list[KBEntry] = []
    skipped = 0

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    df.columns = [str(c).strip() for c in df.columns]

    missing = _KB_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"KB CSV is missing expected columns: {sorted(missing)}")

    # Row index + 2 ~ spreadsheet line number (header is row 1).
    sub = df.loc[:, list(KB_COL_ORDER)]
    for spreadsheet_row_no, (_, row_series) in enumerate(sub.iterrows(), start=2):
        row = {_k: _scalar_to_str(row_series[_k]) for _k in KB_COL_ORDER}
        if not all(row.get(c) for c in ("ticket_id", "subject", "body")):
            logger.warning(
                "KB row %d skipped - empty required field", spreadsheet_row_no
            )
            skipped += 1
            continue

        entries.append(KBEntry(**{k: row[k] for k in KB_COL_ORDER}))

    logger.info(
        "Loaded %d KB entries (%d skipped) from %s", len(entries), skipped, path
    )
    return entries


def load_tickets(path: str | Path) -> list[Ticket]:
    """Parse a ticket JSON file (eval set or custom input) into Ticket objects.

    Each JSON object must have at minimum: ticket_id, subject, body.
    Extra keys (customer_name, plan, expected_*) are read when present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ticket file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw).__name__}")

    tickets: list[Ticket] = []
    skipped = 0

    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            logger.warning(
                "Ticket #%d skipped - expected object, got %s",
                i,
                type(obj).__name__,
            )
            skipped += 1
            continue
        missing = _TICKET_REQUIRED_KEYS - set(obj.keys())
        if missing:
            logger.warning("Ticket #%d skipped - missing keys: %s", i, sorted(missing))
            skipped += 1
            continue

        tickets.append(
            Ticket(
                ticket_id=obj["ticket_id"],
                customer_name=obj.get("customer_name", ""),
                plan=obj.get("plan", ""),
                subject=obj["subject"],
                body=obj["body"],
            )
        )

    logger.info("Loaded %d tickets (%d skipped) from %s", len(tickets), skipped, path)
    return tickets


def inspect_kb(entries: list[KBEntry]) -> dict:
    """Return a summary dict useful for data-exploration logging."""
    categories = Counter(e.category for e in entries)
    priorities = Counter(e.priority for e in entries)
    plans = Counter(e.plan for e in entries)

    if entries:
        body_lens = [len(e.body) for e in entries]
        res_lens = [len(e.resolution) for e in entries]
        avg_body_len = sum(body_lens) / len(body_lens)
        avg_res_len = sum(res_lens) / len(res_lens)
        body_min = min(body_lens)
        body_max = max(body_lens)
        res_min = min(res_lens)
        res_max = max(res_lens)
    else:
        avg_body_len = avg_res_len = 0.0
        body_min = body_max = res_min = res_max = 0

    return {
        "total_entries": len(entries),
        "category_distribution": dict(categories),
        "priority_distribution": dict(priorities),
        "plan_distribution": dict(plans),
        "avg_body_chars": round(avg_body_len),
        "min_body_chars": body_min,
        "max_body_chars": body_max,
        "avg_resolution_chars": round(avg_res_len),
        "min_resolution_chars": res_min,
        "max_resolution_chars": res_max,
    }
