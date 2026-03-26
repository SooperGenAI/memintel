"""
app/models/guardrails_api.py
────────────────────────────────────────────────────────────────────────────────
Pydantic v2 models for the Guardrails API layer.

These models represent versioned guardrails managed via POST /guardrails,
distinct from the file-based Guardrails model in app/models/guardrails.py.

Invariants
──────────
  - Only one GuardrailsVersion is active at any time.
  - Creating a new version supersedes (deactivates) the previous one atomically.
  - Versions are never deleted — only superseded.
  - guardrails_id is auto-generated UUID.
  - version is auto-assigned ("v1", "v2", …) by GuardrailsStore.
  - source tracks whether the guardrails came from the API or file.
  - get_active() never raises — returns None when no DB version exists.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ── GuardrailsDefinition sub-models ───────────────────────────────────────────

class StrategyRegistryEntry(BaseModel):
    """A named strategy in the strategy registry."""
    name: str  # threshold, percentile, z_score, change, equals, composite


class TypeStrategyMapEntry(BaseModel):
    """Maps a primitive type to its valid strategies."""
    type_name: str       # float, int, boolean, string, categorical etc.
    strategies: list[str]


class ParameterPrior(BaseModel):
    """
    Severity-level priors for a signal's threshold parameters.

    Each severity level maps to a dict with at least 'value' and optionally
    'window' (for time-based strategies like change and z_score).
    """
    signal_id: str
    low_severity: dict     # { value: float, window: str | None }
    medium_severity: dict
    high_severity: dict


class BiasRule(BaseModel):
    """A single parameter bias rule mapping a word to a severity level."""
    word: str             # e.g. "urgent", "significant"
    severity_level: str   # high_severity | medium_severity | low_severity


class ThresholdDirection(BaseModel):
    """Direction preference for a signal's threshold strategy."""
    signal_id: str
    direction: Literal["above", "below"]


# ── Top-level GuardrailsDefinition ────────────────────────────────────────────

class GuardrailsDefinition(BaseModel):
    """
    A simplified, API-manageable view of the guardrails configuration.

    strategy_registry lists the names of allowed strategies.
    type_strategy_map maps primitive type names to their valid strategies.
    parameter_priors maps signal IDs to severity-level prior dicts.
    bias_rules maps bias words to severity level strings.
    threshold_directions maps signal IDs to threshold direction strings.
    global_preferred_strategy is the strategy biased toward when no hint exists.
    global_default_strategy is the fallback strategy of last resort.
    """
    strategy_registry: list[str]
    type_strategy_map: dict[str, list[str]]
    parameter_priors: dict[str, dict]
    bias_rules: dict[str, str]
    threshold_directions: dict[str, str] = Field(default_factory=dict)
    global_preferred_strategy: str = "percentile"
    global_default_strategy: str = "threshold"


# ── Request / response models ─────────────────────────────────────────────────

class CreateGuardrailsRequest(BaseModel):
    """Request body for POST /guardrails."""
    guardrails: GuardrailsDefinition
    change_note: str | None = None   # human-readable reason for change


class GuardrailsVersion(BaseModel):
    """
    A versioned snapshot of the guardrails configuration.

    guardrails_id is auto-generated at construction time.
    version is auto-assigned by GuardrailsStore ("v1", "v2", …).
    created_at defaults to the current UTC time.
    is_active is True for the most recently created version.
    source tracks whether this was loaded from the API or from file.
    """
    guardrails_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "v1"          # overwritten by GuardrailsStore.create()
    guardrails: GuardrailsDefinition
    change_note: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_active: bool = True
    source: Literal["api", "file"] = "api"


class GuardrailsImpactResult(BaseModel):
    """
    How tasks are distributed across guardrails versions.

    Returned by GET /guardrails/impact.

    older_version_task_ids lists tasks compiled under a previous guardrails
    version (or no version at all) — these may benefit from re-creation
    under the current guardrails to pick up updated configuration.
    """
    total_tasks: int
    tasks_on_current_version: int
    tasks_on_older_guardrails_version: int
    older_version_task_ids: list[str]
