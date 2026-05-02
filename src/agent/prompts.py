"""
Prompt templates and message assembly for Stage 3 (LLM triage).

Prompt / grounding copy lives here; edit this module for wording only, not for
infra (retries, LiteLLM, concurrency — those belong elsewhere in ``agent``).
"""

from __future__ import annotations

from models import Ticket

# ---------------------------------------------------------------------------
# Category / priority grounding (injected into the system prompt).
# ---------------------------------------------------------------------------
CATEGORY_DESCRIPTIONS = """\
- billing: Invoices, charges, payment methods, plan pricing, refunds, billing page access issues.
- bug: Something that previously worked is now broken — errors, crashes, incorrect behaviour, UI glitches.
- feature_request: Customer wants new functionality or an enhancement that doesn't exist yet.
- account: User management, ownership transfers, permissions, deactivations, password/access issues (non-SSO).
- integration: Third-party connectors (Salesforce, HubSpot, Slack, SSO/SAML, webhooks, API, imports/exports involving external systems).
- onboarding: Getting started, setup guidance, migration from other tools, best-practices questions, documentation lookups.
- security: Data privacy, encryption, compliance, vulnerability reports, audit requirements, 2FA/MFA issues.
- performance: Slowness, high latency, timeouts, resource usage, scaling concerns."""

PRIORITY_DESCRIPTIONS = """\
- low: Informational, nice-to-have, no immediate impact on workflow.
- medium: Causes inconvenience but has a workaround; not blocking core work.
- high: Significantly impacts productivity; no easy workaround; needs attention soon.
- critical: System down, data loss risk, security breach, or large number of users completely blocked."""

# ---------------------------------------------------------------------------
# System prompt (KB context is filled in at call time).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior support triage agent for **Steadfast**, a B2B SaaS project-management platform.

Your job: given a customer support ticket, classify it and draft an initial customer response.

## Categories (pick exactly one)
{categories}

## Priorities (pick exactly one)
{priorities}

## Knowledge Base Context
Short excerpts from resolved tickets. Use only to classify and inform an **immediate** reply. Do NOT copy their wording mechanically; summarise the **action**.

{kb_context}

## Output format
Respond with a single JSON object — no markdown fences, no extra text:

{{
  "reasoning": "<1 short sentence>",
  "category": "<one of the 8 categories>",
  "priority": "<low|medium|high|critical>",
  "response": "<customer-facing reply - specific, actionable>",
  "confidence": <float 0-1>
}}

Rules for ``response``:
- Keep it to **2-3 concise sentences** (roughly 40-60 words).
- Acknowledge the issue briefly (one clause), then provide **specific, actionable next steps**.
- If KB context fits, name the concrete workaround, navigation path, or known resolution; skip generic filler.
- Professional, warm tone. Do NOT invent features or steps absent from KB context."""


def build_system_prompt(kb_chunks: list[str]) -> str:
    """Assemble the full system prompt with KB context."""
    if kb_chunks:
        context_block = "\n\n---\n\n".join(
            f"### Similar Ticket {i + 1}\n{chunk}" for i, chunk in enumerate(kb_chunks)
        )
    else:
        context_block = "(No similar tickets found in the knowledge base.)"

    return SYSTEM_PROMPT.format(
        categories=CATEGORY_DESCRIPTIONS,
        priorities=PRIORITY_DESCRIPTIONS,
        kb_context=context_block,
    )


def build_user_message(ticket: Ticket) -> str:
    """Format the incoming ticket as the user message."""
    parts = [
        f"Ticket ID: {ticket.ticket_id}",
        f"Customer: {ticket.customer_name}" if ticket.customer_name else None,
        f"Plan: {ticket.plan}" if ticket.plan else None,
        f"Subject: {ticket.subject}",
        f"Body:\n{ticket.body}",
    ]
    return "\n".join(p for p in parts if p)
