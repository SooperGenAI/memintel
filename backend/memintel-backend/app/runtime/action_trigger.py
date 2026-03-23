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

Action execution types (current stubs):
  webhook      → would POST to action.config.endpoint (not implemented here).
  notification → would send to action.config.channel (not implemented here).
  workflow     → would trigger action.config.workflow_id (not implemented here).
  register     → would write back to action.config.primitive_id (not implemented).

In this implementation _invoke_action() is a stub that always succeeds
(returns an empty payload).  Real connectors are wired at a higher layer.
Override _invoke_action() in a subclass or inject a dispatcher callable to
connect real delivery backends.
"""
from __future__ import annotations

from typing import Any, Callable

import structlog

from app.models.action import ActionDefinition, FireOn
from app.models.condition import DecisionValue
from app.models.errors import ErrorDetail, ErrorResponse, ErrorType
from app.models.result import ActionTriggered, ActionTriggeredStatus


log = structlog.get_logger(__name__)


class ActionTrigger:
    """
    Triggers the pre-bound actions for a condition decision.

    Parameters
    ──────────
    dispatcher — optional callable(action, decision) → dict used to perform
                 the real delivery (HTTP call, notification, etc.).  When None,
                 trigger_bound_actions() operates in stub mode: 'triggered'
                 status is returned with an empty payload_sent.
    """

    def __init__(
        self,
        dispatcher: Callable[[ActionDefinition, DecisionValue], dict[str, Any]] | None = None,
    ) -> None:
        self._dispatcher = dispatcher

    # ── Public API ─────────────────────────────────────────────────────────────

    def trigger_bound_actions(
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
            triggered = self._process_action(action, decision, dry_run)
            results.append(triggered)

        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_action(
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
            payload = self._invoke_action(action, decision)
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

    def _invoke_action(
        self,
        action: ActionDefinition,
        decision: DecisionValue,
    ) -> dict[str, Any]:
        """
        Invoke the action delivery backend.

        Default implementation: stub that returns an empty payload dict and
        succeeds immediately.  Override or inject a dispatcher for real delivery.
        """
        if self._dispatcher is not None:
            return self._dispatcher(action, decision)

        # Stub: in production, branch on action.config.type and call the
        # appropriate delivery backend (HTTP, notification, workflow, register).
        return {}


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
