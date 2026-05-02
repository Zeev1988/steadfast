"""Single-shot LiteLLM completion + response parsing."""

from __future__ import annotations

from litellm import acompletion

from models import TriageResult

from .config import MAX_OUTPUT_TOKENS, require_model
from .parse_response import parse_response


async def call_and_parse(
    system_prompt: str,
    user_message: str,
    ticket_id: str,
) -> TriageResult:
    """Make a single LLM call via LiteLLM and parse the response.

    Raises ValueError on truncated or unparseable output so Tenacity can retry.
    """
    resp = await acompletion(
        model=require_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)

    # Detect truncation *before* attempting to parse — saves a confusing
    # "no JSON found" error and gives a clear retry signal.
    if finish_reason == "length":
        raise ValueError(
            f"{ticket_id}: output truncated (finish_reason=length, "
            f"max_tokens={MAX_OUTPUT_TOKENS}). Retrying."
        )

    raw_text = choice.message.content or ""
    return parse_response(raw_text, ticket_id)
