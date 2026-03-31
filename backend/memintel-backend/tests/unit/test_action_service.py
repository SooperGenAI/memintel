"""
tests/unit/test_action_service.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ActionService.

Coverage:
  1.  trigger() — fire_on='any'   → status='triggered'
  2.  trigger() — fire_on='true'  → status='triggered' (synthetic value=True)
  3.  trigger() — fire_on='false' → status='skipped'   (True doesn't match)
  4.  trigger() — dry_run=True    → status='would_trigger', no delivery
  5.  trigger() — dispatcher raises → status='failed', error populated
  6.  trigger() — ActionResult fields match ActionTriggered output
  7.  trigger() — returns ActionResult (never raises)
  8.  trigger() — webhook action   → status='triggered'
  9.  trigger() — notification action → status='triggered'
  10. trigger() — workflow action  → status='triggered'
  11. trigger() — register action  → status='triggered'

Test isolation: no DB calls — ActionService.trigger() doesn't access the pool.
ActionTrigger dispatched inline (no mock needed for the stub dispatcher path).
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.action import (
    ActionDefinition,
    ActionResult,
    ActionTriggerRequest,
    FireOn,
    NotificationActionConfig,
    RegisterActionConfig,
    TriggerConfig,
    WebhookActionConfig,
    WorkflowActionConfig,
)
from app.models.result import ActionTriggeredStatus
from app.models.task import Namespace
from app.services.action import ActionService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pool() -> Any:
    """Minimal pool stub — trigger() never touches the pool."""
    return MagicMock()


def _make_service() -> ActionService:
    return ActionService(pool=_pool())


def _run(coro):
    return asyncio.run(coro)


def _make_http_mock(status_code: int = 200) -> Any:
    """Return a mock for httpx.AsyncClient that succeeds without real HTTP."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.request = AsyncMock(return_value=mock_resp)
    return mock_client


def _make_action(
    fire_on: FireOn = FireOn.ANY,
    action_type: str = "register",
    action_id: str = "notify_high_score",
    version: str = "1.0",
) -> ActionDefinition:
    """Build a minimal ActionDefinition for the given fire_on and type."""
    if action_type == "webhook":
        config = WebhookActionConfig(
            type="webhook",
            endpoint="https://example.com/hook",
        )
    elif action_type == "notification":
        config = NotificationActionConfig(
            type="notification",
            channel="slack-alerts",
        )
    elif action_type == "workflow":
        config = WorkflowActionConfig(
            type="workflow",
            workflow_id="wf_escalate",
        )
    else:  # register
        config = RegisterActionConfig(
            type="register",
            primitive_id="org.score_flag",
        )

    trigger = TriggerConfig(
        fire_on=fire_on,
        condition_id="high_score",
        condition_version="1.0",
    )
    return ActionDefinition(
        action_id=action_id,
        version=version,
        config=config,
        trigger=trigger,
        namespace=Namespace.ORG,
    )


def _make_req(
    version: str = "1.0",
    entity: str = "acct_1",
    dry_run: bool = False,
) -> ActionTriggerRequest:
    return ActionTriggerRequest(version=version, entity=entity, dry_run=dry_run)


# ── trigger() tests ────────────────────────────────────────────────────────────

def test_trigger_fire_on_any_triggers():
    """fire_on='any' always fires regardless of the synthetic decision value."""
    action = _make_action(fire_on=FireOn.ANY)
    result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert isinstance(result, ActionResult)
    assert result.status == ActionTriggeredStatus.TRIGGERED
    assert result.action_id == "notify_high_score"
    assert result.action_version == "1.0"


def test_trigger_fire_on_true_triggers():
    """fire_on='true' fires because synthetic decision value is True."""
    action = _make_action(fire_on=FireOn.TRUE)
    result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert result.status == ActionTriggeredStatus.TRIGGERED


def test_trigger_fire_on_false_skips():
    """fire_on='false' skips because synthetic decision value=True != False."""
    action = _make_action(fire_on=FireOn.FALSE)
    result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert result.status == ActionTriggeredStatus.SKIPPED


def test_trigger_dry_run_returns_would_trigger():
    """dry_run=True → status='would_trigger', no real delivery."""
    action = _make_action(fire_on=FireOn.ANY)
    result = _run(_make_service().trigger(action=action, req=_make_req(dry_run=True)))

    assert result.status == ActionTriggeredStatus.WOULD_TRIGGER
    assert result.payload_sent is None
    assert result.error is None


def test_trigger_delivery_failure_returns_failed():
    """When the dispatcher raises, status='failed' with error populated."""
    from app.runtime.action_trigger import ActionTrigger
    from unittest.mock import patch

    def _raising_dispatcher(action, decision):
        raise RuntimeError("Webhook unreachable")

    action = _make_action(fire_on=FireOn.ANY)
    req = _make_req()
    service = _make_service()

    with patch.object(ActionTrigger, "_invoke_action", side_effect=RuntimeError("Webhook unreachable")):
        result = _run(service.trigger(action=action, req=req))

    assert result.status == ActionTriggeredStatus.FAILED
    assert result.error is not None
    assert "Webhook unreachable" in result.error.error.message


def test_trigger_result_fields_populated():
    """ActionResult has correct action_id and action_version."""
    action = _make_action(
        fire_on=FireOn.ANY,
        action_id="my_action",
        version="2.0",
    )
    result = _run(_make_service().trigger(action=action, req=_make_req(version="2.0")))

    assert result.action_id == "my_action"
    assert result.action_version == "2.0"


def test_trigger_never_raises():
    """trigger() must not propagate exceptions — always returns ActionResult."""
    action = _make_action(fire_on=FireOn.ANY)
    # Even with a broken pool, trigger() should not raise
    service = ActionService(pool=None)  # type: ignore[arg-type]
    result = _run(service.trigger(action=action, req=_make_req()))
    assert isinstance(result, ActionResult)


def test_trigger_webhook_action():
    """Webhook action type → triggered successfully (httpx mocked)."""
    action = _make_action(fire_on=FireOn.ANY, action_type="webhook")
    mock_client = _make_http_mock()

    with patch("app.runtime.action_trigger.httpx.AsyncClient", return_value=mock_client):
        result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert result.status == ActionTriggeredStatus.TRIGGERED


def test_trigger_notification_action():
    """Notification action type → triggered successfully via stub dispatcher."""
    action = _make_action(fire_on=FireOn.ANY, action_type="notification")
    result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert result.status == ActionTriggeredStatus.TRIGGERED


def test_trigger_workflow_action():
    """Workflow action type → triggered successfully via stub dispatcher."""
    action = _make_action(fire_on=FireOn.ANY, action_type="workflow")
    result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert result.status == ActionTriggeredStatus.TRIGGERED


def test_trigger_register_action():
    """Register (write-back) action type → triggered via stub dispatcher."""
    action = _make_action(fire_on=FireOn.ANY, action_type="register")
    result = _run(_make_service().trigger(action=action, req=_make_req()))

    assert result.status == ActionTriggeredStatus.TRIGGERED


def test_trigger_dry_run_skipped_fire_on_false():
    """
    When fire_on='false' AND dry_run=True, the fire_on check runs first.
    fire_on='false' skips (no delivery) — dry_run=True is not reached.
    """
    action = _make_action(fire_on=FireOn.FALSE)
    result = _run(_make_service().trigger(action=action, req=_make_req(dry_run=True)))

    # fire_on='false' with synthetic True → skip wins over would_trigger
    assert result.status == ActionTriggeredStatus.SKIPPED
