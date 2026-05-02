"""
Shared enums and Pydantic models for the triage pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(str, Enum):
    billing = "billing"
    bug = "bug"
    feature_request = "feature_request"
    account = "account"
    integration = "integration"
    onboarding = "onboarding"
    security = "security"
    performance = "performance"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


VALID_CATEGORIES: frozenset[str] = frozenset(c.value for c in Category)
VALID_PRIORITIES: frozenset[str] = frozenset(p.value for p in Priority)

# Keys Stage 5 (postprocess) reads from `TriageResult` / ticket pairs.
TRIAGE_KEYS_FOR_POSTPROCESS: frozenset[str] = frozenset(
    {"ticket_id", "category", "priority", "response", "confidence", "flags"}
)


class LlmTriagePayload(BaseModel):
    """Structured shape of LLM JSON after ``json.loads`` (before ``TriageResult``).

    Extra keys are ignored.  ``reasoning`` is optional and stripped downstream.
    Used in ``agent.parse_response.parse_response`` for schema enforcement; invalid payloads
    become retryable ``ValueError``s.
    """

    model_config = ConfigDict(extra="ignore")

    reasoning: str | None = None
    category: str = "unknown"
    priority: str = "medium"
    response: str = ""
    confidence: float | None = None

    @field_validator("category", "priority", "response", mode="before")
    @classmethod
    def _strip_text(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> float | None:
        if v is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, x))

    @model_validator(mode="after")
    def _response_nonempty(self) -> LlmTriagePayload:
        if not self.response:
            raise ValueError("empty or missing response in LLM JSON")
        return self


class Ticket(BaseModel):
    """Normalised representation of an incoming ticket."""

    ticket_id: str
    customer_name: str = ""
    plan: str = ""
    subject: str
    body: str

    @field_validator(
        "ticket_id",
        "customer_name",
        "plan",
        "subject",
        "body",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class KBEntry(BaseModel):
    """One row from the knowledge base CSV."""

    ticket_id: str
    customer_name: str = ""
    plan: str = ""
    subject: str
    body: str
    category: str
    priority: str
    resolution: str = ""

    @field_validator("category", "priority", mode="before")
    @classmethod
    def normalise_lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator(
        "ticket_id",
        "customer_name",
        "plan",
        "subject",
        "body",
        "resolution",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class TriageResult(BaseModel):
    """Pipeline output for a single ticket."""

    ticket_id: str
    category: str
    priority: str
    response: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    # Wall-clock per ticket including retries (Stage 3); optional for tooling.
    processing_seconds: Optional[float] = Field(default=None, ge=0.0)

    def to_dict(self) -> dict:
        out = self.model_dump(exclude_none=True)
        # Always include flags even when empty
        out.setdefault("flags", [])
        return out
