"""LiteLLM / environment wiring for Stage 3 classification."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_BASE_URL = os.environ.get("LLM_BASE_URL")
if _BASE_URL:
    litellm.api_base = _BASE_URL

MODEL: str | None = os.environ.get("LLM_MODEL", None)


def require_model() -> str:
    """Return the configured model string or raise if unset.

    Called at runtime (classify time) rather than import time so that
    test files can import from ``agent`` without needing a ``.env`` file.
    """
    if not MODEL:
        raise ValueError(
            "LLM_MODEL environment variable is not set. "
            "Copy .env.example to .env and set LLM_MODEL."
        )
    return MODEL


MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
