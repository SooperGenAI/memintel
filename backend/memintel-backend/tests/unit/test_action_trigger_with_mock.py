"""
tests/unit/test_action_trigger_with_mock.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ActionTrigger using MockWebhookServer.

These tests exercise ActionTrigger in isolation — no database, no HTTP stack,
no full pipeline.  MockWebhookServer patches httpx.AsyncClient so that
delivery calls are intercepted and inspectable without network access.

Tests
─────
 1. test_webhook_trigger_builds_correct_payload
    Verifies the default decision payload shape when no payload_template is set.

 2. test_webhook_trigger_resolves_env_vars
    Header with ${MY_VAR} placeholder → resolved value sent, not literal string.

 3. test_webhook_trigger_records_failure_on_500
    HTTP 500 → ActionTriggered.status=FAILED, error is non-empty.

 4. test_webhook_trigger_records_success_on_200
    HTTP 200 → ActionTriggered.status=TRIGGERED, payload_sent populated.

 5. test_fire_on_true_only_fires_when_true
    fire_on='true', decision.value=False → no HTTP call, status=SKIPPED.

 6. test_fire_on_false_only_fires_when_false
    fire_on='false', decision.value=True → no HTTP call, status=SKIPPED.

 7. test_fire_on_any_always_fires
    fire_on='any' → fires for both True and False decision values.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.models.action import (
    ActionDefinition,
    FireOn,
    TriggerConfig,
    WebhookActionConfig,
)
from app.models.condition import DecisionType, DecisionValue
from app.models.result import ActionTriggeredStatus
from app.models.task import Namespace
from app.runtime.action_trigger import ActionTrigger
from tests.mocks.mock_webhook_server import MockWebhookServer


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _run(coro: Any) -> Any:
    """Run a coroutine synchronously in a new event loop."""
    return asyncio.run(coro)


def _decision(
    value: bool = True,
    entity: str = "acct_test",
    condition_id: str = "cond_x",
    condition_version: str = "v1",
    timestamp: str = "2025-11-14T09:00:00Z",
) -> DecisionValue:
    return DecisionValue(
        value=value,
        decision_type=DecisionType.BOOLEAN,
        condition_id=condition_id,
        condition_version=condition_version,
        entity=entity,
        timestamp=timestamp,
    )


def _webhook_action(
    fire_on: FireOn = FireOn.ANY,
    endpoint: str = MockWebhookServer.DEFAULT_URL,
    headers: dict | None = None,
    payload_template: dict | None = None,
    action_id: str = "act_1",
    condition_id: str = "cond_x",
    condition_version: str = "v1",
) -> ActionDefinition:
    return ActionDefinition(
        action_id=action_id,
        version="v1",
        config=WebhookActionConfig(
            type="webhook",
            endpoint=endpoint,
            headers=headers or {},
            payload_template=payload_template,
        ),
        trigger=TriggerConfig(
            fire_on=fire_on,
            condition_id=condition_id,
            condition_version=condition_version,
        ),
        namespace=Namespace.ORG,
    )


# ── Tests ────────────────────────────────────────────────────────────────────────

def test_webhook_trigger_builds_correct_payload():
    """
    Default payload (no payload_template) contains all five decision provenance
    fields: entity, value, condition_id, condition_version, timestamp.
    """
    decision = _decision(
        value=True,
        entity="acct_42",
        condition_id="cond_churn",
        condition_version="v2",
        timestamp="2025-11-14T10:00:00Z",
    )
    action = _webhook_action(fire_on=FireOn.ANY)

    server = MockWebhookServer()
    with server:
        triggered = _run(
            ActionTrigger().trigger_bound_actions(
                decision=decision, actions=[action], dry_run=False
            )
        )

    assert server.call_count == 1
    body = server.last_request.json()
    assert body["entity"] == "acct_42"
    assert body["value"] is True
    assert body["condition_id"] == "cond_churn"
    assert body["condition_version"] == "v2"
    assert body["timestamp"] == "2025-11-14T10:00:00Z"

    assert triggered[0].status == ActionTriggeredStatus.TRIGGERED
    assert triggered[0].payload_sent is not None


def test_webhook_trigger_resolves_env_vars(monkeypatch):
    """
    ${MY_VAR} in a header value is replaced with the environment variable value.
    The literal placeholder is never transmitted.
    """
    monkeypatch.setenv("MY_WEBHOOK_TOKEN", "resolved-value-xyz")

    action = _webhook_action(
        fire_on=FireOn.ANY,
        headers={"X-Token": "Bearer ${MY_WEBHOOK_TOKEN}"},
    )

    server = MockWebhookServer()
    with server:
        _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(), actions=[action], dry_run=False
            )
        )

    assert server.call_count == 1
    sent_header = server.last_request.headers.get("X-Token", "")
    assert sent_header == "Bearer resolved-value-xyz", (
        f"Expected resolved value, got: {sent_header!r}"
    )
    assert "${MY_WEBHOOK_TOKEN}" not in sent_header


def test_webhook_trigger_records_failure_on_500():
    """
    HTTP 500 from the endpoint → ActionTriggered.status=FAILED, error populated.
    The exception is never re-raised — the pipeline continues.
    """
    action = _webhook_action(fire_on=FireOn.ANY)

    server = MockWebhookServer(status_code=500)
    with server:
        triggered = _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(), actions=[action], dry_run=False
            )
        )

    assert server.call_count == 1, "delivery was attempted despite failure"
    assert triggered[0].status == ActionTriggeredStatus.FAILED
    assert triggered[0].error is not None
    assert triggered[0].error.error.message  # non-empty error message


def test_webhook_trigger_records_success_on_200():
    """
    HTTP 200 from the endpoint → ActionTriggered.status=TRIGGERED, payload_sent set.
    """
    action = _webhook_action(fire_on=FireOn.ANY)

    server = MockWebhookServer(status_code=200)
    with server:
        triggered = _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(), actions=[action], dry_run=False
            )
        )

    assert server.call_count == 1
    assert triggered[0].status == ActionTriggeredStatus.TRIGGERED
    assert triggered[0].payload_sent is not None
    # payload_sent is the delivery result dict, not the request body.
    assert "endpoint" in triggered[0].payload_sent
    assert triggered[0].payload_sent["http_status"] == 200


def test_fire_on_true_only_fires_when_true():
    """
    fire_on='true': no HTTP call made when decision.value=False; status=SKIPPED.
    """
    action = _webhook_action(fire_on=FireOn.TRUE)

    server = MockWebhookServer()
    with server:
        triggered = _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(value=False), actions=[action], dry_run=False
            )
        )

    assert server.call_count == 0, "no HTTP call when fire_on='true' and decision=False"
    assert triggered[0].status == ActionTriggeredStatus.SKIPPED


def test_fire_on_false_only_fires_when_false():
    """
    fire_on='false': no HTTP call made when decision.value=True; status=SKIPPED.
    """
    action = _webhook_action(fire_on=FireOn.FALSE)

    server = MockWebhookServer()
    with server:
        triggered = _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(value=True), actions=[action], dry_run=False
            )
        )

    assert server.call_count == 0, "no HTTP call when fire_on='false' and decision=True"
    assert triggered[0].status == ActionTriggeredStatus.SKIPPED


def test_fire_on_any_always_fires():
    """
    fire_on='any': webhook fires for both True and False decision values.
    Each call is captured and counted by the mock server.
    """
    action_true  = _webhook_action(fire_on=FireOn.ANY, action_id="act_true")
    action_false = _webhook_action(fire_on=FireOn.ANY, action_id="act_false")

    server = MockWebhookServer()
    with server:
        # fire_on='any' + decision=True → fires
        triggered_true = _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(value=True), actions=[action_true], dry_run=False
            )
        )
        assert server.call_count == 1

        # fire_on='any' + decision=False → also fires
        triggered_false = _run(
            ActionTrigger().trigger_bound_actions(
                decision=_decision(value=False), actions=[action_false], dry_run=False
            )
        )
        assert server.call_count == 2

    assert triggered_true[0].status  == ActionTriggeredStatus.TRIGGERED
    assert triggered_false[0].status == ActionTriggeredStatus.TRIGGERED
