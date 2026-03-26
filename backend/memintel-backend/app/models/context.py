"""
app/models/context.py
────────────────────────────────────────────────────────────────────────────────
Pydantic v2 models for the Application Context layer.

ApplicationContext captures four dimensions of domain knowledge that guide
LLM-driven task authoring:
  domain        — what the application does, what entities it tracks, what
                  decisions it needs to make
  behavioural   — how data arrives (batch/streaming), useful time windows,
                  regulatory constraints
  semantic_hints — domain-specific vocabulary to inject into LLM prompts
  calibration_bias — cost asymmetry between false negatives and false positives,
                     used to shift strategy priors at task creation time

Invariants
──────────
  - Only one ApplicationContext version is active at any time.
  - Creating a new context supersedes (deactivates) the previous one atomically.
  - Context is never deleted — only superseded.
  - context_id is auto-generated UUID.
  - version is auto-assigned ("v1", "v2", …) by ContextStore.
  - bias_direction is always auto-derived from false_negative_cost vs
    false_positive_cost. Any caller-supplied value is silently overwritten.
  - get_active() never raises — returns None when no context exists.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Cost ordering helper ───────────────────────────────────────────────────────

_COST_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


# ── Sub-models ────────────────────────────────────────────────────────────────

class EntityDeclaration(BaseModel):
    """A named entity type the application tracks."""
    name: str
    description: str


class SemanticHint(BaseModel):
    """Domain-specific vocabulary entry injected into LLM prompts."""
    term: str
    definition: str


class CalibrationBias(BaseModel):
    """
    Cost asymmetry between false negatives and false positives.

    bias_direction is always auto-derived — never set by the caller:
      false_negative_cost > false_positive_cost  → recall
        (missing a real event is more costly; prefer sensitivity)
      false_positive_cost > false_negative_cost  → precision
        (spurious alerts are more costly; prefer specificity)
      equal costs                                → balanced
    """
    false_negative_cost: Literal["high", "medium", "low"]
    false_positive_cost: Literal["high", "medium", "low"]
    bias_direction: Literal["recall", "precision", "balanced"] = "balanced"

    @model_validator(mode="after")
    def _derive_bias_direction(self) -> "CalibrationBias":
        """Always overwrite bias_direction from the two cost fields."""
        fn = _COST_RANK[self.false_negative_cost]
        fp = _COST_RANK[self.false_positive_cost]
        if fn > fp:
            self.bias_direction = "recall"
        elif fp > fn:
            self.bias_direction = "precision"
        else:
            self.bias_direction = "balanced"
        return self


class DomainContext(BaseModel):
    """What the application does, what entities it tracks, what decisions it makes."""
    description: str
    entities: list[EntityDeclaration] = []
    decisions: list[str] = []


class BehaviouralContext(BaseModel):
    """How data arrives and any operational constraints."""
    data_cadence: Literal["batch", "streaming", "mixed"] = "batch"
    meaningful_windows: dict | None = None
    regulatory: list[str] = []


# ── Top-level context model ───────────────────────────────────────────────────

class ApplicationContext(BaseModel):
    """
    A versioned snapshot of the application's domain context.

    context_id is auto-generated at construction time.
    version is auto-assigned by ContextStore ("v1", "v2", …).
    created_at defaults to the current UTC time.
    is_active is True for the most recently created context.
    """
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "v1"          # overwritten by ContextStore.create()
    domain: DomainContext
    behavioural: BehaviouralContext = Field(default_factory=BehaviouralContext)
    semantic_hints: list[SemanticHint] = []
    calibration_bias: CalibrationBias | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_active: bool = True


# ── Request / response models ─────────────────────────────────────────────────

class CreateContextRequest(BaseModel):
    """
    Request body for POST /context.

    bias_direction inside calibration_bias (if supplied) is ignored —
    it is always auto-derived by CalibrationBias.model_validator.
    """
    domain: DomainContext
    behavioural: BehaviouralContext = Field(default_factory=BehaviouralContext)
    semantic_hints: list[SemanticHint] = []
    calibration_bias: CalibrationBias | None = None


class ContextImpactResult(BaseModel):
    """
    How tasks are distributed across context versions.

    Returned by GET /context/impact.

    older_version_task_ids lists tasks compiled under a previous context
    version (or no context at all) — these may benefit from re-creation
    under the current context to pick up updated domain knowledge.
    """
    total_tasks: int
    tasks_on_current_version: int
    tasks_on_older_versions: int
    older_version_task_ids: list[str]
