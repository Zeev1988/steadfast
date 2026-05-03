"""
Exploratory summary for the knowledge base CSV and eval ticket JSON.

Run::

    PYTHONPATH=src python -m loader.eda_kb_eval

Default log path: ``<project>/output/eda_kb_eval.log``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from loader import inspect_kb, load_knowledge_base, load_tickets

_MULTI_ISSUE_PAT = re.compile(
    r"multiple\s+issues|several\s+problems|\d+\)|first[,;:].*second",
    re.IGNORECASE,
)
_INTEGRATION_PAT = re.compile(
    r"slack|jira|github|okta|salesforce|api|webhook|sso",
    re.IGNORECASE,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def build_report(
    kb_path: Path,
    eval_path: Path,
) -> str:
    lines: list[str] = []

    def out(msg: str = "") -> None:
        lines.append(msg)

    kb_entries = load_knowledge_base(kb_path)
    tickets = load_tickets(eval_path)
    with eval_path.open(encoding="utf-8") as fh:
        ev_raw = json.load(fh)
    ev = pd.DataFrame(ev_raw)
    kb_df = pd.read_csv(kb_path, dtype=str, keep_default_na=False)

    out("=== Steadfast KB + Eval EDA ===")
    out(f"KB file: {kb_path}")
    out(f"Eval file: {eval_path}")
    out()

    out("=== Knowledge base (loader-parsed) ===")
    summary = inspect_kb(kb_entries)
    out(f"Entries loaded: {summary['total_entries']} (CSV rows: {len(kb_df)})")
    out(f"Category distribution: {summary['category_distribution']}")
    out(f"Priority distribution: {summary['priority_distribution']}")
    out(f"Plan distribution: {summary['plan_distribution']}")
    out(
        f"Avg body chars: {summary['avg_body_chars']} "
        f"(min/max {summary['min_body_chars']}/{summary['max_body_chars']})"
    )
    out(
        f"Avg resolution chars: {summary['avg_resolution_chars']} "
        f"(min/max {summary['min_resolution_chars']}/{summary['max_resolution_chars']})"
    )

    empty_br = sum(
        1
        for e in kb_entries
        if not (e.body or "").strip() and not (e.resolution or "").strip()
    )
    out(f"Entries with empty body and resolution: {empty_br}")
    out()

    out("=== Eval set ===")
    out(f"Tickets: {len(tickets)}")
    if "expected_category" in ev.columns:
        out("expected_category:\n" + ev["expected_category"].value_counts().to_string())
        out()
    if "expected_priority" in ev.columns:
        out("expected_priority:\n" + ev["expected_priority"].value_counts().to_string())
        out()
    if "plan" in ev.columns:
        out("plan:\n" + ev["plan"].value_counts().to_string())
        out()

    for col in ("subject", "body"):
        if col in ev.columns:
            L = ev[col].astype(str).str.len()
            out(
                f"Eval {col} chars: min={int(L.min())} p50={L.median():.0f} "
                f"mean={L.mean():.1f} max={int(L.max())}"
            )
    out()

    kb_cat = pd.Series([e.category for e in kb_entries]).value_counts(normalize=True)
    ev_cat = ev["expected_category"].value_counts(normalize=True)
    cmp = pd.DataFrame({"kb_frac": kb_cat, "eval_frac": ev_cat}).fillna(0)
    cmp["delta_eval_minus_kb"] = cmp["eval_frac"] - cmp["kb_frac"]
    out("=== Category fraction (eval - kb); positive = over-represented in eval ===")
    out(cmp.sort_values("delta_eval_minus_kb", ascending=False).to_string())
    out()

    kb_pri = pd.Series([e.priority for e in kb_entries]).value_counts(normalize=True)
    ev_pri = ev["expected_priority"].value_counts(normalize=True)
    pri_cmp = pd.DataFrame({"kb_frac": kb_pri, "eval_frac": ev_pri}).fillna(0)
    out("=== Priority priors (fractions) ===")
    out(pri_cmp.to_string())
    out()

    kb_lens = pd.Series([len(e.body) for e in kb_entries])
    p95 = kb_lens.quantile(0.95)
    body_lens = ev["body"].astype(str).str.len()
    long_idx = body_lens > float(p95)
    out(f"=== Eval bodies longer than KB body p95 ({p95:.0f} chars): {int(long_idx.sum())} ===")
    for _, r in ev.loc[long_idx].iterrows():
        out(
            f"  {r['ticket_id']}: {len(r['body'])} chars | "
            f"{r['expected_category']}/{r['expected_priority']}"
        )
    out()

    text_ev = ev["subject"].astype(str) + " " + ev["body"].astype(str)
    multi = text_ev.str.contains(_MULTI_ISSUE_PAT)
    out(f"Eval multi-issue heuristic matches: {int(multi.sum())} / {len(ev)}")
    if multi.any():
        out(ev.loc[multi, ["ticket_id", "expected_category", "expected_priority"]].to_string(index=False))
    out()

    kb_text = kb_df["subject"].astype(str) + " " + kb_df["body"].astype(str)
    kb_i = kb_text.str.contains(_INTEGRATION_PAT).mean()
    ev_i = text_ev.str.contains(_INTEGRATION_PAT).mean()
    out(f"Share body+subject matching integration-ish regex: KB {kb_i:.3f}, eval {ev_i:.3f}")
    out()

    kb_e = (kb_df["plan"].astype(str).str.lower() == "enterprise").mean()
    ev_e = (ev["plan"].astype(str).str.lower() == "enterprise").mean()
    out(f"Enterprise plan share: KB {kb_e:.3f}, eval {ev_e:.3f}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    p = argparse.ArgumentParser(description="KB + eval EDA → log file")
    p.add_argument(
        "--kb",
        type=Path,
        default=root / "data" / "knowledge_base.csv",
    )
    p.add_argument(
        "--eval",
        type=Path,
        default=root / "data" / "eval_set.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="eda_kb_eval.log",
        help="Filename under output-dir",
    )
    args = p.parse_args(argv)

    if not args.kb.exists():
        print(f"KB not found: {args.kb}", file=sys.stderr)
        return 1
    if not args.eval.exists():
        print(f"Eval not found: {args.eval}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / args.log_file

    report = build_report(args.kb, args.eval)
    log_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[eda_kb_eval] Wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
