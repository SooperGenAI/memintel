"""
app/models/task.py
──────────────────────────────────────────────────────────────────────────────
Task domain models.

Covers the full lifecycle of a task: creation request, internal representation,
update request, and paginated list response.

Design notes
────────────
Task — single model serves both the DB row and the API response shape.
  Fields that exist in the DB but are NOT part of the API contract
  (version for optimistic locking, updated_at) are excluded from JSON
  serialisation via Field(exclude=True). They remain fully accessible as
  Python attributes for the store layer.

  Rationale: a separate TaskRow / TaskResponse split would require a
  mapping step in every store method. The exclude approach keeps the type
  signature clean while respecting the API contract.

task_id — optional (None) at construction time because the DB sets it via
  gen_random_uuid(). TaskStore.create() returns the populated record.

DeliveryConfig — validates type-specific required sub-fields at model
  construction time so that invalid configs are rejected before they reach
  the DB (fail fast at the request boundary).

IMMUTABLE_TASK_FIELDS — a frozenset consumed by TaskStore.update() to
  reject changes to fields that are permanently pinned at creation.
  Defined here so it lives next to the model it guards.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class DeliveryType(str, Enum):
    WEBHOOK      = "webhook"
    NOTIFICATION = "notification"
    EMAIL        = "email"
    WORKFLOW     = "workflow"


class TaskStatus(str, Enum):
    """
    Full set of task statuses.

    active  — evaluated by the application on its own schedule.
    paused  — skipped; no evaluation or alerts until resumed.
    deleted — soft-deleted; retained for audit, no evaluation. Irreversible via API.
    preview — dry_run result only; MUST NEVER be persisted to the DB.
    """
    ACTIVE  = "active"
    PAUSED  = "paused"
    DELETED = "deleted"
    PREVIEW = "preview"


class MutableTaskStatus(str, Enum):
    """
    Subset of TaskStatus that a PATCH request is allowed to set.

    Deletion is a dedicated DELETE endpoint, not a status update.
    preview cannot be set via PATCH — it is only produced by dry_run.
    """
    ACTIVE = "active"
    PAUSED = "paused"


class Namespace(str, Enum):
    PERSONAL = "personal"
    TEAM     = "team"
    ORG      = "org"
    GLOBAL   = "global"


class Sensitivity(str, Enum):
    """Hint to the LLM for threshold prior selection (used in ConstraintsConfig)."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Immutability contract ─────────────────────────────────────────────────────

#: Fields on Task that must never change after creation.
#: Consumed by TaskStore.update() — any update dict containing one of these
#: keys must be rejected with HTTP 400 before touching the DB.
#:
#: condition_version is intentionally excluded: it is mutable and is the
#: mechanism by which a task is rebound to a new calibrated condition version.
IMMUTABLE_TASK_FIELDS: frozenset[str] = frozenset({
    "concept_id",
    "concept_version",
    "condition_id",
    "action_id",
    "action_version",
})


# ── V7 deferred import ────────────────────────────────────────────────────────
# concept.py imports Namespace from this module. To avoid a circular import,
# VocabularyContext must be imported AFTER Namespace is defined above.
# With `from __future__ import annotations`, all annotations in this file are
# lazy strings; Pydantic resolves them at model-use time, by which point both
# modules are fully loaded and VocabularyContext is in this module's namespace.
from app.models.concept import VocabularyContext  # noqa: E402


# ── V7 reasoning trace models ─────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """
    A single step in the Chain of Reasoning (CoR) trace.

    Produced by both POST /tasks (agent CoR) and POST /concepts/compile
    (concept CoR). Each step records what was evaluated, what alternatives
    were considered, and whether the step was accepted, skipped, or failed.

    outcome values:
      accepted — step completed and produced a result.
      skipped  — step was intentionally bypassed (e.g. concept_id shortcut
                 skips Step 2; the step still appears in the trace with
                 outcome='skipped' and an explanatory summary).
      failed   — step could not complete; the pipeline aborts.
    """
    step_index: int
    label:      str
    summary:    str
    candidates: list[str] | None = None   # alternatives considered at this step
    outcome:    Literal["accepted", "skipped", "failed"]
    # Structured decision data — populated by the pipeline, absent when step was skipped
    concept_id:    str   | None = None   # Step 2: concept selected
    output_type:   str   | None = None   # Step 2: concept output type
    strategy_type: str   | None = None   # Step 3: strategy chosen (e.g. "threshold")
    direction:     str   | None = None   # Step 3: above/below/top/bottom/any
    threshold:     float | None = None   # Step 3: numeric cutoff (threshold/z_score only)
    channel:       str   | None = None   # Step 4: delivery channel (notification/webhook/…)
    action_type:   str   | None = None   # Step 4: action execution type


class ReasoningTrace(BaseModel):
    """
    Full Chain of Reasoning trace for a task compilation or concept compilation.

    Returned in the POST /tasks response when return_reasoning=True, and in
    the POST /concepts/compile response when return_reasoning=True.

    compilation_duration_ms — wall-clock time for the full CoR pipeline.
      None when the pipeline was aborted (a step failed before completion).
    """
    steps:                   list[ReasoningStep]
    compilation_duration_ms: int | None = None


# ── DeliveryConfig ────────────────────────────────────────────────────────────

class DeliveryConfig(BaseModel):
    """
    Describes how a triggered condition delivers its alert.

    Validation rules (enforced at model construction time):
      webhook      → endpoint is required
      notification → channel is required
      email        → channel is required
      workflow     → workflow_id is required
    """
    type: DeliveryType
    endpoint: str | None = None     # required when type=webhook
    channel: str | None = None      # required when type=notification or email
    workflow_id: str | None = None  # required when type=workflow

    @model_validator(mode="after")
    def _check_type_specific_fields(self) -> DeliveryConfig:
        if self.type == DeliveryType.WEBHOOK and not self.endpoint:
            raise ValueError("endpoint is required when delivery type is 'webhook'")
        if self.type in (DeliveryType.NOTIFICATION, DeliveryType.EMAIL) and not self.channel:
            raise ValueError(
                f"channel is required when delivery type is '{self.type.value}'"
            )
        if self.type == DeliveryType.WORKFLOW and not self.workflow_id:
            raise ValueError("workflow_id is required when delivery type is 'workflow'")
        return self


# ── ConstraintsConfig ─────────────────────────────────────────────────────────

class ConstraintsConfig(BaseModel):
    """
    Optional authoring-time constraints forwarded to the LLM during POST /tasks.

    sensitivity — overrides the severity derived from natural language intent.
      The LLM uses this as a prior for threshold selection within the guardrails
      bounds. Does not bypass guardrails — only shifts the prior within them.

    namespace — the registry namespace in which the generated definitions will
      be registered. Defaults to 'personal' if not supplied.
    """
    sensitivity: Sensitivity | None = None
    namespace: Namespace | None = None


# ── Task ──────────────────────────────────────────────────────────────────────

class Task(BaseModel):
    """
    Canonical task model. Represents both the DB row and the API response.

    Lifecycle
    ─────────
    Creation  → task_id is None; populated by TaskStore.create() from DB default.
    Read      → all fields populated from the DB row.
    Update    → TaskStore.update() enforces IMMUTABLE_TASK_FIELDS and optimistic
                locking via the version field.
    Delete    → status transitions to 'deleted'; the row is never hard-deleted.
    Preview   → status='preview'; TaskStore.create() must reject these rows.

    Internal fields (excluded from JSON serialisation)
    ──────────────────────────────────────────────────
    version    — optimistic locking counter. Read before UPDATE, increment in
                 WHERE clause. Never exposed to API callers.
    updated_at — DB-managed timestamp. Not in the API spec; internal only.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    task_id: str | None = None       # None until the DB assigns it on insert
    intent: str                      # original natural-language intent, immutable

    # ── Version-pinned references (all immutable after creation) ──────────────
    concept_id: str
    concept_version: str
    condition_id: str
    condition_version: str           # mutable: rebind via PATCH with new version
    action_id: str
    action_version: str

    # ── Operational fields (mutable via PATCH) ────────────────────────────────
    entity_scope: str
    delivery: DeliveryConfig
    status: TaskStatus = TaskStatus.ACTIVE

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime | None = None
    last_triggered_at: datetime | None = None

    # ── Context tracking ──────────────────────────────────────────────────────
    # context_version — the active ApplicationContext version at task creation
    # time. NULL when no context existed at creation. Stored in the DB.
    # context_warning — informational only; populated on creation when no
    # context was defined. Not stored in DB; not present on subsequent reads.
    context_version: str | None = None
    context_warning: str | None = None

    # ── Guardrails tracking ────────────────────────────────────────────────────
    # guardrails_version — the active API guardrails version at task creation
    # time. NULL when no API guardrails version was active (file-based only).
    # Stored in the DB.
    guardrails_version: str | None = None

    # ── V7 reasoning trace ────────────────────────────────────────────────────
    # Populated at task creation time when return_reasoning=True in the request.
    # Never stored in the DB — absent on all subsequent reads (GET /tasks/{id}).
    # The route handler must omit this field entirely (not return null) when
    # return_reasoning=False. See M-5 for response_model_exclude_none handling.
    reasoning_trace: ReasoningTrace | None = None

    # ── Internal DB fields — excluded from API serialisation ──────────────────
    # These fields are written to / read from the DB by the store layer.
    # They must never appear in HTTP responses.
    updated_at: datetime | None = Field(default=None, exclude=True)
    version: int = Field(default=1, exclude=True)

    model_config = {"populate_by_name": True}


# ── Request / response models ─────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    """
    Request body for POST /tasks.

    intent, entity_scope, and delivery are required.
    constraints is optional — when absent, the LLM uses guardrail defaults.
    dry_run=True returns a DryRunResult (see models/result.py) without
    persisting the task.

    V7 additions (all optional with defaults — existing callers unaffected):
      concept_id         — bind to a pre-compiled concept registered via
                           POST /concepts/register. When provided, CoR Step 2
                           (Concept Selection) is skipped; steps 1, 3, 4 still run.
      vocabulary_context — bound CoR Step 2 to the provided concept/condition IDs.
                           Three states: absent=global fallback, empty lists=
                           vocabulary_mismatch (service-layer error), non-empty=bounded.
      stream             — when True, response is text/event-stream (SSE). The
                           synchronous JSON path is used when False (default).
      return_reasoning   — when True, reasoning_trace is populated in the response.
                           When False (default), reasoning_trace is absent entirely.
    """
    intent: str
    entity_scope: str
    delivery: DeliveryConfig
    constraints: ConstraintsConfig | None = None
    dry_run: bool = False

    # V7 optional fields — all have defaults so existing callers are unaffected
    concept_id:         str | None = None
    vocabulary_context: VocabularyContext | None = None
    stream:             bool = False
    return_reasoning:   bool = False


class TaskUpdateRequest(BaseModel):
    """
    Request body for PATCH /tasks/{id}.

    All fields are optional, but at least one must be provided.
    The validator raises a parameter_error if the request is empty.

    Allowed changes:
      condition_version — rebind to a new version of the same condition
      delivery          — change delivery channel/endpoint/etc.
      entity_scope      — change the evaluated entity or group
      status            — pause (paused) or resume (active) only

    Forbidden changes (rejected by TaskStore.update() with HTTP 400):
      concept_id, concept_version, condition_id, action_id, action_version
      These require creating a new task via POST /tasks.
    """
    condition_version: str | None = None
    delivery: DeliveryConfig | None = None
    entity_scope: str | None = None
    status: MutableTaskStatus | None = None

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> TaskUpdateRequest:
        provided = {
            k for k, v in self.model_dump().items() if v is not None
        }
        if not provided:
            raise ValueError(
                "At least one of condition_version, delivery, entity_scope, "
                "or status must be provided."
            )
        return self

    def to_patch_dict(self) -> dict[str, Any]:
        """
        Returns a dict of only the fields that were explicitly set.

        Used by TaskStore.update() to build a targeted UPDATE statement.
        Excludes None values — a None field means "not provided", not "set to null".
        """
        return {
            k: v
            for k, v in self.model_dump().items()
            if v is not None
        }


class TaskList(BaseModel):
    """
    Paginated task list response for GET /tasks.

    Pagination is cursor-based. next_cursor is the task_id of the last item
    in the current page — pass it as ?cursor= on the next request.
    next_cursor is None when has_more is False.
    """
    items: list[Task]
    has_more: bool
    next_cursor: str | None = None
    total_count: int
