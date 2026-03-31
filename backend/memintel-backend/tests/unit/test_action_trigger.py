"""
tests/unit/test_action_trigger.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ActionTrigger._invoke_action() backend implementations.

Coverage:
  1.  notification — no CANVAS_NOTIFICATION_ENDPOINT → status='logged'
  2.  notification — CANVAS_NOTIFICATION_ENDPOINT set → HTTP POST, status='sent'
  3.  notification — custom message_template is substituted correctly
  4.  webhook — HTTP POST → status='sent', endpoint in payload
  5.  webhook — ${ENV_VAR} in headers resolved from environment
  6.  webhook — HTTP 4xx raises so _process_action records status='failed'
  7.  workflow — no CANVAS_WORKFLOW_ENDPOINT → status='skipped'
  8.  register — always returns skipped with primitive_id, no HTTP call
"""
from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.action import (
    ActionDefinition,
    FireOn,
    NotificationActionConfig,
    RegisterActionConfig,
    TriggerConfig,
    WebhookActionConfig,
    WorkflowActionConfig,
)
from app.models.condition import DecisionType, DecisionValue
from app.models.result import ActionTriggeredStatus
from app.models.task import Namespace
from app.runtime.action_trigger import ActionTrigger


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decision(entity: str = "acct_1") -> DecisionValue:
    return DecisionValue(
        value=True,
        decision_type=DecisionType.BOOLEAN,
        condition_id="cond_x",
        condition_version="1.0",
        entity=entity,
        timestamp="2025-01-15T12:00:00Z",
    )


def _trigger_cfg() -> TriggerConfig:
    return TriggerConfig(
        fire_on=FireOn.ANY,
        condition_id="cond_x",
        condition_version="1.0",
    )


def _notification_action(message_template: str | None = None) -> ActionDefinition:
    return ActionDefinition(
        action_id="notify_1",
        version="1.0",
        config=NotificationActionConfig(
            type="notification",
            channel="slack-alerts",
            message_template=message_template,
        ),
        trigger=_trigger_cfg(),
        namespace=Namespace.ORG,
    )


def _webhook_action(
    endpoint: str = "https://example.com/hook",
    headers: dict[str, str] | None = None,
) -> ActionDefinition:
    return ActionDefinition(
        action_id="webhook_1",
        version="1.0",
        config=WebhookActionConfig(
            type="webhook",
            endpoint=endpoint,
            headers=headers or {},
        ),
        trigger=_trigger_cfg(),
        namespace=Namespace.ORG,
    )


def _workflow_action() -> ActionDefinition:
    return ActionDefinition(
        action_id="wf_1",
        version="1.0",
        config=WorkflowActionConfig(
            type="workflow",
            workflow_id="wf_escalate",
            input_mapping={"entity": "entity"},
        ),
        trigger=_trigger_cfg(),
        namespace=Namespace.ORG,
    )


def _register_action() -> ActionDefinition:
    return ActionDefinition(
        action_id="reg_1",
        version="1.0",
        config=RegisterActionConfig(
            type="register",
            primitive_id="org.score_flag",
        ),
        trigger=_trigger_cfg(),
        namespace=Namespace.ORG,
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _make_http_mock(status_code: int = 200) -> tuple[Any, Any]:
    """Return (mock_client, mock_response) with async context manager wired up."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.request = AsyncMock(return_value=mock_response)

    return mock_client, mock_response


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_notification_no_endpoint_returns_logged(monkeypatch):
    """Without CANVAS_NOTIFICATION_ENDPOINT, notification is logged (not POSTed)."""
    monkeypatch.delenv("CANVAS_NOTIFICATION_ENDPOINT", raising=False)

    trigger = ActionTrigger()
    result = _run(trigger._invoke_action(_notification_action(), _decision()))

    assert result["status"] == "logged"
    assert result["channel"] == "slack-alerts"
    assert "message" in result
    assert isinstance(result["message"], str)
    assert len(result["message"]) > 0


def test_notification_with_endpoint_posts_and_returns_sent(monkeypatch):
    """When CANVAS_NOTIFICATION_ENDPOINT is set, a POST is made and status='sent'."""
    monkeypatch.setenv("CANVAS_NOTIFICATION_ENDPOINT", "https://canvas.internal/notify")
    mock_client, mock_response = _make_http_mock(status_code=200)

    with patch("app.runtime.action_trigger.httpx.AsyncClient", return_value=mock_client):
        trigger = ActionTrigger()
        result = _run(trigger._invoke_action(_notification_action(), _decision()))

    assert result["status"] == "sent"
    assert result["channel"] == "slack-alerts"
    assert result["http_status"] == 200
    mock_client.post.assert_awaited_once()


def test_notification_custom_template_substituted(monkeypatch):
    """message_template is formatted using decision payload fields."""
    monkeypatch.delenv("CANVAS_NOTIFICATION_ENDPOINT", raising=False)

    template = "Entity {entity} fired condition {condition_id}"
    trigger = ActionTrigger()
    result = _run(trigger._invoke_action(
        _notification_action(message_template=template),
        _decision(entity="acct_42"),
    ))

    assert result["status"] == "logged"
    assert "acct_42" in result["message"]
    assert "cond_x" in result["message"]


def test_webhook_posts_and_returns_sent():
    """Webhook action POSTs to the configured endpoint and returns status='sent'."""
    mock_client, mock_response = _make_http_mock(status_code=201)

    with patch("app.runtime.action_trigger.httpx.AsyncClient", return_value=mock_client):
        trigger = ActionTrigger()
        result = _run(trigger._invoke_action(_webhook_action(), _decision()))

    assert result["status"] == "sent"
    assert result["endpoint"] == "https://example.com/hook"
    assert result["http_status"] == 201
    mock_client.request.assert_awaited_once()


def test_webhook_resolves_env_var_in_headers(monkeypatch):
    """${ENV_VAR} placeholders in header values are replaced with env values."""
    monkeypatch.setenv("WEBHOOK_TOKEN", "secret-token-xyz")
    mock_client, _ = _make_http_mock()

    action = _webhook_action(headers={"Authorization": "Bearer ${WEBHOOK_TOKEN}"})

    with patch("app.runtime.action_trigger.httpx.AsyncClient", return_value=mock_client):
        trigger = ActionTrigger()
        _run(trigger._invoke_action(action, _decision()))

    # Verify the actual request used the resolved header value.
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer secret-token-xyz"


def test_webhook_http_error_propagates():
    """HTTP 4xx raises so _process_action records status='failed'."""
    import httpx as _httpx

    mock_client, mock_response = _make_http_mock(status_code=400)
    mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
        "400 Bad Request",
        request=MagicMock(),
        response=MagicMock(),
    )

    with patch("app.runtime.action_trigger.httpx.AsyncClient", return_value=mock_client):
        trigger = ActionTrigger()
        with pytest.raises(_httpx.HTTPStatusError):
            _run(trigger._invoke_action(_webhook_action(), _decision()))


def test_workflow_no_endpoint_returns_skipped(monkeypatch):
    """Without CANVAS_WORKFLOW_ENDPOINT, workflow action is skipped non-fatally."""
    monkeypatch.delenv("CANVAS_WORKFLOW_ENDPOINT", raising=False)

    trigger = ActionTrigger()
    result = _run(trigger._invoke_action(_workflow_action(), _decision()))

    assert result["status"] == "skipped"
    assert "CANVAS_WORKFLOW_ENDPOINT" in result["reason"]
    assert result["workflow_id"] == "wf_escalate"


def test_register_returns_skipped_with_primitive_id():
    """Register action always returns skipped (write-back not yet implemented)."""
    trigger = ActionTrigger()
    result = _run(trigger._invoke_action(_register_action(), _decision()))

    assert result["status"] == "skipped"
    assert result["primitive_id"] == "org.score_flag"
    assert "not yet implemented" in result["reason"]
