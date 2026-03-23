"""
app/models/action.py
──────────────────────────────────────────────────────────────────────────────
Action domain models for the α (alpha) layer.

Covers the full action lifecycle:
  - ActionType / FireOn enums
  - Per-type config models (WebhookActionConfig, NotificationActionConfig,
    WorkflowActionConfig, RegisterActionConfig) — discriminated union
  - TriggerConfig — fire_on rule + condition binding
  - ActionDefinition — registered, versioned action definition
  - ActionTriggerRequest — request body for POST /actions/{id}/trigger
  - ActionResult — response from POST /actions/{id}/trigger
  - ActionList — paginated list response for GET /actions

Design notes
────────────
ActionConfig is a discriminated union keyed on the `type` field, mirroring
  the StrategyDefinition pattern in condition.py. The type is embedded inside
  the config variant — ActionDefinition has no separate `type` field.
  The executor accesses action.config.type for dispatch.

TriggerConfig carries both the fire_on rule and the condition binding
  (condition_id + condition_version). This encapsulates the complete
  'when does this action fire?' question in one object. The executor
  calls _should_fire(action.trigger.fire_on, decision.value) to check.

FireOn string values are 'true' / 'false' / 'any' (lowercase strings) —
  NOT Python booleans. The enum uses TRUE/FALSE names to avoid shadowing
  Python's built-in True/False, but the wire values are lowercase strings.

Actions are best-effort — at-most-once per evaluation call, no automatic
  retry. A failed action is recorded in ActionTriggeredStatus='failed';
  the pipeline returns HTTP 200 regardless. Downstream systems are
  responsible for their own idempotency using the deduplication key:
    (condition_id, condition_version, entity, timestamp)

Action execution types (from py-instructions.md):
  webhook      → HTTP POST to an external endpoint
  notification → push notification via channel
  workflow     → trigger an external workflow engine
  register     → write the decision result back to a primitive (write-back)

RegisterActionConfig.primitive_id names the primitive that receives the
  write-back value. This enables closed-loop feedback patterns where a
  decision outcome updates an observable primitive.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.models.errors import ErrorResponse
from app.models.result import ActionTriggeredStatus
from app.models.task import Namespace


# ── Enums ─────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """
    Registered action execution types.

    webhook      — HTTP POST to an external URL; payload_template controls body
    notification — push notification via a named channel
    workflow     — trigger an external workflow engine
    register     — write the decision result back to a primitive (write-back loop)

    Note: 'email' is a DeliveryType in task.py (Task delivery config), not an
    action execution type. Actions use these four types only.
    """
    WEBHOOK      = "webhook"
    NOTIFICATION = "notification"
    WORKFLOW     = "workflow"
    REGISTER     = "register"


class FireOn(str, Enum):
    """
    Controls when a bound action fires relative to the condition decision.

    TRUE  — fires when decision.value is True (or the matched label for equals)
    FALSE — fires when decision.value is False
    ANY   — fires on every evaluation, regardless of decision value

    Wire values are lowercase strings: 'true', 'false', 'any'.
    These are NOT Python booleans — do not compare with `is True` / `is False`.
    """
    TRUE  = "true"
    FALSE = "false"
    ANY   = "any"


# ── Per-type action config models ─────────────────────────────────────────────
# Each variant carries a Literal `type` field used as the Pydantic v2
# discriminator. Wire format: {"type": "webhook", "endpoint": "...", ...}

class WebhookActionConfig(BaseModel):
    """
    Config for webhook actions. The executor POSTs to `endpoint` when fired.

    payload_template is an opaque dict; the executor fills entity, timestamp,
    decision_value, and condition_id into it before sending. When absent, the
    executor sends a default payload containing all decision provenance fields.

    headers may include auth tokens — these should reference ${ENV_VAR}
    substitutions resolved at config apply time, never logged as plaintext.

    method defaults to 'POST'. 'GET' is permitted for fire-and-forget pings.
    """
    type: Literal[ActionType.WEBHOOK]
    endpoint: str                                      # required: target URL
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    payload_template: dict[str, Any] | None = None    # None → default payload


class NotificationActionConfig(BaseModel):
    """
    Config for push notification actions.

    channel is the named notification channel (e.g. 'slack-alerts', 'pagerduty').
    message_template is an optional format string; the executor interpolates
    entity, decision_value, and condition_id. When absent, a default message
    containing the decision summary is used.
    """
    type: Literal[ActionType.NOTIFICATION]
    channel: str
    message_template: str | None = None


class WorkflowActionConfig(BaseModel):
    """
    Config for workflow trigger actions.

    workflow_id references a registered workflow in the connected workflow engine.
    input_mapping maps workflow input parameter names to decision fields:
      key   — workflow engine input name
      value — source field from the decision payload (e.g. 'entity', 'value')
    When empty, the executor forwards the full decision payload as-is.
    """
    type: Literal[ActionType.WORKFLOW]
    workflow_id: str
    input_mapping: dict[str, str] = Field(default_factory=dict)


class RegisterActionConfig(BaseModel):
    """
    Config for write-back (register) actions.

    Writes the decision result back to a named primitive in the Memintel
    primitive registry. Enables closed-loop patterns: a decision outcome
    updates an observable signal that future concept evaluations can consume.

    primitive_id — the primitive to update (format: namespace.field)
    entity_field — which field from the decision payload carries the entity ID
                   that identifies which primitive record to update.
                   Defaults to 'entity' (decision.entity).
    """
    type: Literal[ActionType.REGISTER]
    primitive_id: str
    entity_field: str = "entity"


#: Discriminated union — the single type used for ActionDefinition.config.
#: Pydantic v2 resolves the correct variant from the 'type' field at parse time.
ActionConfig = Annotated[
    WebhookActionConfig
    | NotificationActionConfig
    | WorkflowActionConfig
    | RegisterActionConfig,
    Field(discriminator="type"),
]


# ── TriggerConfig ─────────────────────────────────────────────────────────────

class TriggerConfig(BaseModel):
    """
    Firing rule + condition binding for a registered action.

    fire_on determines when the action fires relative to the decision value:
      'true'  → fires when decision=True (or matched label for equals strategy)
      'false' → fires when decision=False
      'any'   → fires on every evaluation regardless of decision value

    condition_id + condition_version pin this action to a specific, immutable
    condition version. Bound at action registration time; never resolved at
    execution time (no dynamic action resolution).

    The executor evaluates _should_fire(trigger.fire_on, decision.value)
    before invoking the action config.
    """
    fire_on: FireOn
    condition_id: str
    condition_version: str


# ── ActionDefinition ──────────────────────────────────────────────────────────

class ActionDefinition(BaseModel):
    """
    A registered, versioned action definition. The α layer record.

    Stored in the `definitions` table with definition_type='action'.
    The body JSONB column holds the serialisation of this model.

    Immutability: once registered, an (action_id, version) pair is permanent.
    Updates require a new version. The registry enforces this at the DB level.

    The execution type and all delivery parameters are inside config.
    Access config.type for dispatch: WebhookActionConfig, NotificationActionConfig,
    WorkflowActionConfig, or RegisterActionConfig.

    Actions are pre-bound at task creation time. There is no runtime action
    resolution — the executor fires only the pre-bound action for this record.
    """
    action_id: str
    version: str
    config: ActionConfig
    trigger: TriggerConfig
    namespace: Namespace
    created_at: datetime | None = None
    deprecated: bool = False


# ── ActionTriggerRequest ───────────────────────────────────────────────────────

class ActionTriggerRequest(BaseModel):
    """
    Request body for POST /actions/{id}/trigger.

    Triggers a registered action directly for a given entity, bypassing the
    full pipeline. Used for:
      - Verifying action config before go-live
      - Staging environment testing
      - Manual trigger for debugging

    dry_run=True simulates the trigger without making any HTTP calls or
    external side effects. The response has status='would_trigger'.

    version is required — actions are always addressed by explicit (id, version).
    There is no 'latest' resolution.
    """
    version: str
    entity: str
    timestamp: str | None = None
    dry_run: bool = False


# ── ActionResult ──────────────────────────────────────────────────────────────

class ActionResult(BaseModel):
    """
    Response from POST /actions/{id}/trigger.

    Distinct from ActionTriggered (result.py) which is nested inside
    DecisionResult and produced by the pipeline. ActionResult is the
    top-level response for the direct-trigger endpoint.

    status mirrors ActionTriggeredStatus:
      triggered    — invoked and delivery succeeded
      skipped      — fire_on rule not met (only possible if a decision value
                     is provided and does not match fire_on)
      failed       — invoked but delivery raised an error
      would_trigger — dry_run=True; action was not invoked

    payload_sent is the actual payload delivered (populated when triggered).
    error is populated only when status='failed'.
    """
    action_id: str
    action_version: str
    status: ActionTriggeredStatus
    payload_sent: dict[str, Any] | None = None
    error: ErrorResponse | None = None


# ── ActionList ────────────────────────────────────────────────────────────────

class ActionList(BaseModel):
    """
    Paginated action list response for GET /actions.

    Cursor-based pagination. next_cursor is the action_id of the last item
    in the current page — pass it as ?cursor= on the next request.
    next_cursor is None when has_more is False.
    """
    items: list[ActionDefinition]
    has_more: bool
    next_cursor: str | None = None
    total_count: int
