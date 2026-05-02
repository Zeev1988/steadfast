"""Extract JSON from raw LLM text and validate into TriageResult."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from models import LlmTriagePayload, TriageResult

# Match the first { ... } block (handles nested braces one level deep)
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_response(raw_text: str, ticket_id: str) -> TriageResult:
    """Extract JSON from the LLM response and build a TriageResult.

    Raises ValueError on parse failure so the caller can retry.
    """
    match = _JSON_RE.search(raw_text)
    if not match:
        raise ValueError(f"No JSON object found in LLM response for {ticket_id}")

    data = json.loads(match.group())

    try:
        payload = LlmTriagePayload.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM JSON schema invalid for {ticket_id}: {exc}") from exc

    category = payload.category.strip().lower()
    priority = payload.priority.strip().lower()

    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        response=payload.response,
        confidence=payload.confidence,
    )
