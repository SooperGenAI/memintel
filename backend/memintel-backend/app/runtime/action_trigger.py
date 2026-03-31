"""
app/runtime/action_trigger.py
──────────────────────────────────────────────────────────────────────────────
ActionTrigger — executes the bound actions for a condition decision.

Best-effort contract (from py-instructions.md and core-spec.md §1C):
  - MUST NOT block the pipeline response.
  - The pipeline returns HTTP 200 regardless of individual action outcomes.
  - Failures are captured in ActionTriggered.status='failed'; never re-raised.
  - At-most-once per evaluation call — no automatic retry on failure.
  - dry_run=True: status='would_trigger', NO HTTP call, no side effects.

fire_on semantics:
  'true'  → fires when decision.value is True (bool) or a matched label (str).
  'false' → fires when decision.value is False.
  'any'   → fires on every evaluation regardless of decision value.

Action execution types:
  webhook      → async POST to action.config.endpoint via httpx.
  notification → async POST to CANVAS_NOTIFICATION_ENDPOINT if set; else log.
  workflow     → async POST to CANVAS_WORKFLOW_ENDPOINT if set; else skip.
  register     → log + return skipped (write-back not yet implemented).

${ENV_VAR} substitutions are resolved in webhook header values at call time.
They are NOT resolved in endpoint URLs — endpoints are treated as literals.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

import httpx
import structlog

from app.models.action import ActionDefinition, ActionType, FireOn
from app.models.condition import DecisionValue
from app.models.errors import ErrorDetail, ErrorResponse, ErrorType
from app.models.result import ActionTriggered, ActionTriggeredStatus


log = structlog.get_logger(__name__)

# Default message template for notification actions when message_template is absent.
_DEFAULT_NOTIFICATION_TEMPLATE = (
    "Condition {condition_id} fired for {entity} — value {value} at {timestamp}"
)

# Regex for ${ENV_VAR} substitution in webhook header values.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_vars(headers: dict[str, str]) -> dict[str, str]:
    """
    Resolve ``${ENV_VAR}`` placeholders in header values.

    Only header VALUES are substituted — keys and endpoint URLs are untouched.
    Unset variables are left as-is (the placeholder remains in the value).
    """
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved[key] = _ENV_VAR_RE.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    return resolved


class _SafeFormat(dict):  # type: ignore[type-arg]
    """Format-map that leaves unknown placeholders unchanged."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


class ActionTrigger:
    """
    Triggers the pre-bound actions for a condition decision.

    Parameters
    ──────────
    dispatcher — optional callable(action, decision) → dict used to perform
                 the real delivery (HTTP call, notification, etc.).  When None,
                 trigger_bound_actions() dispatches to the built-in backends.
    """

    def __init__(
        self,
        dispatcher: Callable[[ActionDefinition, DecisionValue], dict[str, Any]] | None = None,
    ) -> None:
        self._dispatcher = dispatcher

    # ── Public API ─────────────────────────────────────────────────────────────

    async def trigger_bound_actions(
        self,
        decision: DecisionValue,
        actions: list[ActionDefinition],
        dry_run: bool = False,
    ) -> list[ActionTriggered]:
        """
        Evaluate and fire each action in ``actions`` based on the decision.

        For every action:
          1. If fire_on rule is not met                 → status='skipped'
          2. If dry_run=True                            → status='would_trigger'
          3. If _invoke_action() succeeds               → status='triggered'
          4. If _invoke_action() raises any exception   → status='failed'

        Exceptions from delivery are NEVER re-raised.  The pipeline continues
        regardless of individual action outcomes (best-effort contract).

        At-most-once: each action is attempted exactly once.  No retry.
        """
        results: list[ActionTriggered] = []

        for action in actions:
            triggered = await self._process_action(action, decision, dry_run)
            results.append(triggered)

        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _process_action(
        self,
        action: ActionDefinition,
        decision: DecisionValue,
        dry_run: bool,
    ) -> ActionTriggered:
        """Process a single action for the given decision."""
        if not _should_fire(action.trigger.fire_on, decision.value):
            log.info(
                "action_skipped",
                action_id=action.action_id,
                condition_id=decision.condition_id,
                fire_on=action.trigger.fire_on.value,
            )
            return ActionTriggered(
                action_id=action.action_id,
                action_version=action.version,
                status=ActionTriggeredStatus.SKIPPED,
            )

        if dry_run:
            log.info(
                "action_would_trigger",
                action_id=action.action_id,
                condition_id=decision.condition_id,
            )
            return ActionTriggered(
                action_id=action.action_id,
                action_version=action.version,
                status=ActionTriggeredStatus.WOULD_TRIGGER,
            )

        # Attempt delivery — best-effort, never block pipeline on failure.
        try:
            payload = await self._invoke_action(action, decision)
            log.info(
                "action_triggered",
                action_id=action.action_id,
                condition_id=decision.condition_id,
            )
            return ActionTriggered(
                action_id=action.action_id,
                action_version=action.version,
                status=ActionTriggeredStatus.TRIGGERED,
                payload_sent=payload,
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "action_failed",
                action_id=action.action_id,
                condition_id=decision.condition_id,
                error=str(exc),
            )
            return ActionTriggered(
                action_id=action.action_id,
                action_version=action.version,
                status=ActionTriggeredStatus.FAILED,
                error=ErrorResponse(
                    error=ErrorDetail(
                        type=ErrorType.EXECUTION_ERROR,
                        message=str(exc),
                    )
                ),
            )

    async def _invoke_action(
        self,
        action: ActionDefinition,
        decision: DecisionValue,
    ) -> dict[str, Any]:
        """
        Invoke the action delivery backend for the given action type.

        Dispatches to the injected dispatcher if present; otherwise routes
        to the built-in backend for each ActionType variant.

        All return values are dicts (never None).  Failures raise — the
        caller (_process_action) catches and records status='failed'.
        """
        if self._dispatcher is not None:
            return self._dispatcher(action, decision)

        # Build the canonical decision payload sent to all backends.
        decision_payload: dict[str, Any] = {
            "entity": decision.entity,
            "value": decision.value,
            "condition_id": decision.condition_id,
            "condition_version": decision.condition_version,
            "timestamp": decision.timestamp,
        }

        config = action.config
        action_type = config.type

        if action_type == ActionType.NOTIFICATION:
            return await _invoke_notification(config, decision_payload)  # type: ignore[arg-type]

        if action_type == ActionType.WEBHOOK:
            return await _invoke_webhook(config, decision_payload)  # type: ignore[arg-type]

        if action_type == ActionType.WORKFLOW:
            return await _invoke_workflow(config, decision_payload)  # type: ignore[arg-type]

        if action_type == ActionType.REGISTER:
            return _invoke_register(config)  # type: ignore[arg-type]

        # Unknown type — log and succeed silently so pipeline is never blocked.
        log.warning("action_type_unknown", action_type=action_type)
        return {}


# ── Backend implementations ────────────────────────────────────────────────────

async def _invoke_notification(
    config: Any,  # NotificationActionConfig
    decision_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Deliver a notification action.

    If CANVAS_NOTIFICATION_ENDPOINT is set, POST the message payload to it.
    Otherwise log at INFO level and return {"status": "logged"}.
    """
    template = config.message_template or _DEFAULT_NOTIFICATION_TEMPLATE
    message = template.format_map(_SafeFormat(decision_payload))

    notification_payload: dict[str, Any] = {
        "channel": config.channel,
        "message": message,
        **decision_payload,
    }

    endpoint = os.environ.get("CANVAS_NOTIFICATION_ENDPOINT")
    if not endpoint:
        log.info(
            "notification_logged",
            channel=config.channel,
            message=message,
            entity=decision_payload.get("entity"),
        )
        return {
            "status": "logged",
            "channel": config.channel,
            "message": message,
        }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(endpoint, json=notification_payload)
        resp.raise_for_status()

    return {
        "status": "sent",
        "channel": config.channel,
        "http_status": resp.status_code,
    }


async def _invoke_webhook(
    config: Any,  # WebhookActionConfig
    decision_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Deliver a webhook action via HTTP POST (or GET).

    Header values with ``${ENV_VAR}`` placeholders are resolved from the
    environment at call time.  The endpoint URL is used verbatim.

    HTTP 4xx/5xx responses raise httpx.HTTPStatusError — the caller records
    status='failed'.  Connection/timeout errors also propagate as exceptions.
    """
    headers = _resolve_env_vars(config.headers)
    body = config.payload_template if config.payload_template is not None else decision_payload

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.request(
            method=config.method,
            url=config.endpoint,
            json=body,
            headers=headers,
        )
        resp.raise_for_status()

    return {
        "status": "sent",
        "endpoint": config.endpoint,
        "http_status": resp.status_code,
    }


async def _invoke_workflow(
    config: Any,  # WorkflowActionConfig
    decision_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Trigger an external workflow via CANVAS_WORKFLOW_ENDPOINT.

    If the env var is not set, the action is skipped (non-fatal).
    POSTs ``{"workflow_id", "input", "input_mapping"}`` to the endpoint.
    """
    endpoint = os.environ.get("CANVAS_WORKFLOW_ENDPOINT")
    if not endpoint:
        log.info(
            "workflow_skipped_no_endpoint",
            workflow_id=config.workflow_id,
        )
        return {
            "status": "skipped",
            "reason": "CANVAS_WORKFLOW_ENDPOINT not configured",
            "workflow_id": config.workflow_id,
        }

    workflow_payload: dict[str, Any] = {
        "workflow_id": config.workflow_id,
        "input": decision_payload,
        "input_mapping": config.input_mapping,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(endpoint, json=workflow_payload)
        resp.raise_for_status()

    return {
        "status": "triggered",
        "workflow_id": config.workflow_id,
        "http_status": resp.status_code,
    }


def _invoke_register(
    config: Any,  # RegisterActionConfig
) -> dict[str, Any]:
    """
    Write-back (register) action — not yet implemented.

    TODO: implement primitive write-back via the connector registry.
    """
    log.info(
        "register_action_skipped",
        primitive_id=config.primitive_id,
        reason="register write-back not yet implemented",
    )
    return {
        "status": "skipped",
        "reason": "register write-back not yet implemented",
        "primitive_id": config.primitive_id,
    }


# ── Fire-on evaluation ────────────────────────────────────────────────────────

def _should_fire(fire_on: FireOn, decision_value: bool | str) -> bool:
    """
    Return True when the action's fire_on rule matches the decision value.

    fire_on='any'   → always fires.
    fire_on='true'  → fires when decision_value is True (bool) or a non-empty
                      string (matched label for equals strategy).
    fire_on='false' → fires when decision_value is False (bool).
    """
    if fire_on == FireOn.ANY:
        return True
    if fire_on == FireOn.TRUE:
        if isinstance(decision_value, bool):
            return decision_value is True
        # Categorical (equals strategy): non-empty matched label counts as 'true'.
        return bool(decision_value)
    if fire_on == FireOn.FALSE:
        if isinstance(decision_value, bool):
            return decision_value is False
        # Categorical: empty string (no match) counts as 'false'.
        return not bool(decision_value)
    return False
