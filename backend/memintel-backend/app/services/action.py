"""
app/services/action.py
──────────────────────────────────────────────────────────────────────────────
ActionService — action dispatch and execution.

Dispatches webhook, notification, workflow, and register actions.
Evaluates fire_on conditions, handles dry_run mode, and wraps delivery
failures in ActionResult rather than raising exceptions.

Actions are best-effort — trigger() always returns ActionResult and never
raises, even on delivery failure.

Direct-trigger semantics (POST /actions/{id}/trigger)
───────────────────────────────────────────────────────
When trigger() is called directly (bypassing the pipeline), no real condition
evaluation has occurred. We construct a synthetic DecisionValue representing
an unconditional trigger (value=True, fire_on='any' effectively). The action
fire_on rule is still evaluated:
  - fire_on='any'   → always fires
  - fire_on='true'  → fires (True is the synthetic value)
  - fire_on='false' → skipped (False doesn't match True)

This models the route docstring: "the action is fired unconditionally unless
the action executor determines a skip condition from the trigger config."

ActionTrigger handles the actual dispatch and captures errors in ActionResult.
"""
from __future__ import annotations

from typing import Any

import asyncpg
import structlog

from app.models.action import ActionDefinition, ActionResult, ActionTriggerRequest
from app.models.condition import DecisionType, DecisionValue
from app.models.result import ActionTriggered, ActionTriggeredStatus
from app.runtime.action_trigger import ActionTrigger

log = structlog.get_logger(__name__)


class ActionService:
    """
    Dispatches registered actions for a given entity.

    trigger() evaluates fire_on, handles dry_run simulation, and executes
    the delivery mechanism (webhook POST, notification, workflow enqueue,
    or entity registration). Failures are captured in ActionResult.status
    and ActionResult.error — never raised as exceptions.

    Actions are best-effort — trigger() always returns an ActionResult.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def trigger(
        self,
        action: ActionDefinition,
        req: ActionTriggerRequest,
    ) -> ActionResult:
        """
        Fire a registered action directly for a given entity.

        Bypasses the full ψ → φ → α pipeline. Constructs a synthetic
        DecisionValue with value=True (unconditional trigger) and passes
        it to ActionTrigger for fire_on evaluation and dispatch.

        dry_run=True simulates the trigger without making HTTP calls or
        producing external side effects. Returns status='would_trigger'.

        Always returns ActionResult — never raises.
        """
        # Build a synthetic DecisionValue for the direct-trigger path.
        # value=True means fire_on='true' and fire_on='any' both fire;
        # fire_on='false' skips. This honours the trigger config while
        # treating the direct-trigger as an unconditional positive signal.
        decision = DecisionValue(
            value=True,
            decision_type=DecisionType.BOOLEAN,
            condition_id=action.trigger.condition_id,
            condition_version=action.trigger.condition_version,
            entity=req.entity,
        )

        trigger_engine = ActionTrigger()
        results = await trigger_engine.trigger_bound_actions(
            decision=decision,
            actions=[action],
            dry_run=req.dry_run,
        )

        # trigger_bound_actions always returns one result per action.
        triggered: ActionTriggered = results[0]

        log.info(
            "action_service_trigger",
            action_id=action.action_id,
            version=req.version,
            entity=req.entity,
            dry_run=req.dry_run,
            status=triggered.status,
        )

        return ActionResult(
            action_id=triggered.action_id,
            action_version=triggered.action_version,
            status=triggered.status,
            payload_sent=triggered.payload_sent,
            error=triggered.error,
        )
