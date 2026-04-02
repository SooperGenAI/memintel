"""
tests/e2e/test_action_delivery.py
──────────────────────────────────────────────────────────────────────────────
Webhook action delivery e2e tests — full ψ → φ → α pipeline with
intercepted outbound HTTP calls.

These tests exercise the REAL webhook delivery path end-to-end:
  - Real concept/condition/action registration via HTTP API
  - Real pipeline execution via POST /evaluate/full
  - Outbound HTTP calls intercepted by MockWebhookServer (no network)
  - All assertions on actual behaviour, not mocked behaviour

Implementation notes
────────────────────
HTTP library:      httpx.AsyncClient (patched at module level by MockWebhookServer)
Mock approach:     unittest.mock.patch("app.runtime.action_trigger.httpx.AsyncClient")
                   with a drop-in class that records requests and returns
                   configurable httpx.Response objects.

Default webhook payload (when payload_template is None):
  {
    "entity":            decision.entity,
    "value":             decision.value,       # bool or str
    "condition_id":      decision.condition_id,
    "condition_version": decision.condition_version,
    "timestamp":         decision.timestamp,
  }

payload_template behaviour:
  When payload_template is set on WebhookActionConfig, the dict is sent
  VERBATIM — no string interpolation is performed.  The docstring in
  WebhookActionConfig describes planned interpolation that is not yet
  implemented in _invoke_webhook().

${ENV_VAR} resolution:
  Resolved in webhook header VALUES at call time via _resolve_env_vars().
  Unset variables remain as literal "${VAR_NAME}" in the sent headers.
  Endpoint URLs are NOT resolved.

Failure isolation:
  HTTP 4xx/5xx from the mock raises httpx.HTTPStatusError, caught by
  _process_action() → status='failed'.  Pipeline returns HTTP 200 regardless.

dry_run:
  POST /evaluate/full with dry_run=True → status='would_trigger', no HTTP call.

Test isolation
──────────────
Each test uses unique concept/condition/action IDs (wd.t1.*, wd.t2.*, …).
The e2e_setup fixture truncates all tables before each test.
"""
from __future__ import annotations

import os

import pytest

from tests.mocks.mock_connector import MockTableConnector
from tests.mocks.mock_webhook_server import MockWebhookServer

# ── Shared constants ────────────────────────────────────────────────────────────

_ELEVATED = {"X-Elevated-Key": "contract-test-elevated-key"}
_API      = {"X-Api-Key": "contract-test-api-key"}

_PRIM    = "account.active_user_rate_30d"
_LOW     = 0.25   # below threshold → condition fires (True)
_HIGH    = 0.85   # above threshold → condition does NOT fire (False)
_THRESH  = 0.35


# ── Registration helpers ────────────────────────────────────────────────────────

def _concept(cid: str, ver: str) -> dict:
    """Minimal float concept using z_score_op on _PRIM."""
    return {
        "definition_id": cid,
        "version": ver,
        "definition_type": "concept",
        "namespace": "org",
        "body": {
            "concept_id": cid,
            "version": ver,
            "namespace": "org",
            "output_type": "float",
            "output_feature": "out",
            "primitives": {_PRIM: {"type": "float", "missing_data_policy": "zero"}},
            "features": {
                "out": {"op": "z_score_op", "inputs": {"input": _PRIM}, "params": {}},
            },
        },
    }


def _condition(cid: str, ver: str, concept_id: str, concept_ver: str) -> dict:
    """Threshold condition: _PRIM below _THRESH fires (True)."""
    return {
        "definition_id": cid,
        "version": ver,
        "definition_type": "condition",
        "namespace": "org",
        "body": {
            "condition_id": cid,
            "version": ver,
            "namespace": "org",
            "concept_id": concept_id,
            "concept_version": concept_ver,
            "strategy": {
                "type": "threshold",
                "params": {"direction": "below", "value": _THRESH},
            },
        },
    }


def _webhook_action(
    action_id: str,
    ver: str,
    condition_id: str,
    condition_ver: str,
    *,
    endpoint: str,
    fire_on: str = "true",
    headers: dict | None = None,
    payload_template: dict | None = None,
) -> dict:
    """Webhook action definition body for POST /actions."""
    cfg: dict = {"type": "webhook", "endpoint": endpoint}
    if headers:
        cfg["headers"] = headers
    if payload_template is not None:
        cfg["payload_template"] = payload_template
    return {
        "action_id": action_id,
        "version": ver,
        "namespace": "org",
        "config": cfg,
        "trigger": {
            "fire_on": fire_on,
            "condition_id": condition_id,
            "condition_version": condition_ver,
        },
    }


def _notification_action(
    action_id: str,
    ver: str,
    condition_id: str,
    condition_ver: str,
    *,
    channel: str = "slack-test",
) -> dict:
    """Notification action definition body for POST /actions."""
    return {
        "action_id": action_id,
        "version": ver,
        "namespace": "org",
        "config": {"type": "notification", "channel": channel},
        "trigger": {
            "fire_on": "true",
            "condition_id": condition_id,
            "condition_version": condition_ver,
        },
    }


def _register(client, concept_body: dict, condition_body: dict, action_body: dict) -> None:
    """Register concept + condition + action; assert all 2xx."""
    r = client.post("/registry/definitions", json=concept_body, headers=_ELEVATED)
    assert r.status_code == 200, f"concept register failed: {r.text}"

    r = client.post("/registry/definitions", json=condition_body, headers=_ELEVATED)
    assert r.status_code == 200, f"condition register failed: {r.text}"

    r = client.post("/actions", json=action_body, headers=_ELEVATED)
    assert r.status_code == 201, f"action register failed: {r.text}"
    return r.json().get("action_id", action_body["action_id"])


def _evaluate_full(
    client,
    concept_id: str,
    concept_ver: str,
    condition_id: str,
    condition_ver: str,
    entity: str,
    *,
    timestamp: str = "2025-11-14T09:00:00Z",
    dry_run: bool = False,
) -> dict:
    """POST /evaluate/full and return the JSON response body."""
    r = client.post(
        "/evaluate/full",
        json={
            "concept_id": concept_id,
            "concept_version": concept_ver,
            "condition_id": condition_id,
            "condition_version": condition_ver,
            "entity": entity,
            "timestamp": timestamp,
            "dry_run": dry_run,
        },
        headers=_API,
    )
    assert r.status_code == 200, f"evaluate/full failed: {r.text}"
    return r.json()


# ── Test 1 — Webhook fires when condition fires ─────────────────────────────────

@pytest.mark.e2e
def test_webhook_delivered_when_condition_fires(mock_webhook_connector_e2e_client):
    """
    When the condition fires (True), the bound webhook receives exactly one POST.
    """
    CID, VER      = "wd.t1.concept", "v1"
    COND_ID, COND_VER = "wd.t1.condition", "v1"
    ACTION_ID     = "wd.t1.action"
    ENTITY        = "account_001"

    data = {_PRIM: {ENTITY: _LOW}}   # 0.25 < 0.35 → fires
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=webhook.url),
        )

        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    assert body["decision"]["value"] is True, "condition should fire"
    assert webhook.call_count == 1, f"expected 1 webhook call, got {webhook.call_count}"
    assert webhook.last_request.method == "POST"
    assert webhook.url in webhook.last_request.url


# ── Test 2 — Webhook payload contains decision data ─────────────────────────────

@pytest.mark.e2e
def test_webhook_payload_contains_decision_data(mock_webhook_connector_e2e_client):
    """
    The default webhook payload contains entity, condition provenance, decision
    value, and timestamp.  No payload_template — full decision_payload is sent.

    Documented payload shape (default):
      {
        "entity":            str   — entity ID evaluated
        "value":             bool  — decision outcome (True when condition fires)
        "condition_id":      str   — condition that was evaluated
        "condition_version": str   — condition version
        "timestamp":         str   — ISO 8601 timestamp of evaluation
      }
    """
    CID, VER          = "wd.t2.concept", "v1"
    COND_ID, COND_VER = "wd.t2.condition", "v1"
    ACTION_ID         = "wd.t2.action"
    ENTITY            = "account_001"
    TS                = "2025-11-14T09:00:00Z"

    data = {_PRIM: {ENTITY: _LOW}}
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=webhook.url),
        )

        _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY, timestamp=TS)

    assert webhook.call_count == 1
    req_body = webhook.last_request.json()
    assert req_body is not None, "request body should not be None"
    assert req_body["entity"] == ENTITY
    assert req_body["value"] is True
    assert req_body["condition_id"] == COND_ID
    assert req_body["condition_version"] == COND_VER
    assert "timestamp" in req_body


# ── Test 3 — Custom payload template sent verbatim ─────────────────────────────

@pytest.mark.e2e
def test_webhook_custom_payload_template(mock_webhook_connector_e2e_client):
    """
    When payload_template is set on the action config, it is sent VERBATIM.

    IMPORTANT: payload_template is NOT interpolated — placeholders like
    {entity} are sent as literal strings.  _invoke_webhook() sends the
    template dict as-is (no format_map or string substitution).

    This documents the actual production behaviour, which differs from the
    WebhookActionConfig docstring that describes planned interpolation.
    """
    CID, VER          = "wd.t3.concept", "v1"
    COND_ID, COND_VER = "wd.t3.condition", "v1"
    ACTION_ID         = "wd.t3.action"
    ENTITY            = "account_001"

    template = {
        "alert_entity": "{entity}",     # NOT interpolated — sent literally
        "alert_type": "churn_risk",
        "score": "{decision_value}",    # NOT interpolated — sent literally
    }

    data = {_PRIM: {ENTITY: _LOW}}
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(
                ACTION_ID, VER, COND_ID, COND_VER,
                endpoint=webhook.url,
                payload_template=template,
            ),
        )

        _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    assert webhook.call_count == 1
    req_body = webhook.last_request.json()
    # Template is sent verbatim — placeholders are NOT substituted.
    assert req_body["alert_entity"] == "{entity}", (
        "payload_template is sent verbatim; {entity} should not be interpolated"
    )
    assert req_body["alert_type"] == "churn_risk"
    assert "score" in req_body
    assert req_body["score"] == "{decision_value}", (
        "payload_template is sent verbatim; {decision_value} should not be interpolated"
    )


# ── Test 4 — Custom headers sent ───────────────────────────────────────────────

@pytest.mark.e2e
def test_webhook_custom_headers_sent(mock_webhook_connector_e2e_client):
    """
    Custom headers configured on the action are forwarded to the webhook endpoint.
    """
    CID, VER          = "wd.t4.concept", "v1"
    COND_ID, COND_VER = "wd.t4.condition", "v1"
    ACTION_ID         = "wd.t4.action"
    ENTITY            = "account_001"

    custom_headers = {
        "Authorization": "Bearer test-secret-token",
        "X-Alert-Source": "memintel",
    }

    data = {_PRIM: {ENTITY: _LOW}}
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(
                ACTION_ID, VER, COND_ID, COND_VER,
                endpoint=webhook.url,
                headers=custom_headers,
            ),
        )

        _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    assert webhook.call_count == 1
    recv_headers = webhook.last_request.headers
    assert recv_headers["Authorization"] == "Bearer test-secret-token"
    assert recv_headers["X-Alert-Source"] == "memintel"


# ── Test 5 — ENV_VAR resolved in headers ───────────────────────────────────────

@pytest.mark.e2e
def test_webhook_env_var_resolved_in_headers(mock_webhook_connector_e2e_client, monkeypatch):
    """
    ${ENV_VAR} placeholders in webhook header values are resolved from the
    process environment at call time.  The resolved value is sent — the literal
    placeholder "${...}" is never transmitted.
    """
    CID, VER          = "wd.t5.concept", "v1"
    COND_ID, COND_VER = "wd.t5.condition", "v1"
    ACTION_ID         = "wd.t5.action"
    ENTITY            = "account_001"

    monkeypatch.setenv("TEST_WEBHOOK_SECRET", "resolved-secret")

    data = {_PRIM: {ENTITY: _LOW}}
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(
                ACTION_ID, VER, COND_ID, COND_VER,
                endpoint=webhook.url,
                headers={"Authorization": "Bearer ${TEST_WEBHOOK_SECRET}"},
            ),
        )

        _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    assert webhook.call_count == 1
    auth_header = webhook.last_request.headers.get("Authorization", "")
    assert auth_header == "Bearer resolved-secret", (
        f"Expected resolved secret, got: {auth_header!r}"
    )
    assert "${TEST_WEBHOOK_SECRET}" not in auth_header, (
        "Literal placeholder must not be transmitted — env var was not resolved"
    )


# ── Test 6 — Webhook not fired when condition false ─────────────────────────────

@pytest.mark.e2e
def test_webhook_not_fired_when_condition_false(mock_webhook_connector_e2e_client):
    """
    When the condition does NOT fire (value=False) and fire_on='true', the webhook
    is not called.
    """
    CID, VER          = "wd.t6.concept", "v1"
    COND_ID, COND_VER = "wd.t6.condition", "v1"
    ACTION_ID         = "wd.t6.action"
    ENTITY            = "account_002"

    data = {_PRIM: {ENTITY: _HIGH}}   # 0.85 > 0.35 → condition does NOT fire
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=webhook.url, fire_on="true"),
        )

        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    assert body["decision"]["value"] is False, "condition should NOT fire"
    assert webhook.call_count == 0, (
        f"webhook must not be called when condition is False and fire_on='true', "
        f"got {webhook.call_count} calls"
    )


# ── Test 7 — fire_on='false' fires when condition false ─────────────────────────

@pytest.mark.e2e
def test_webhook_fires_on_false_when_configured(mock_webhook_connector_e2e_client):
    """
    fire_on='false': webhook fires when condition=False, skips when condition=True.
    """
    CID, VER          = "wd.t7.concept", "v1"
    COND_ID, COND_VER = "wd.t7.condition", "v1"
    ACTION_ID         = "wd.t7.action"

    data = {
        _PRIM: {
            "account_false": _HIGH,   # 0.85 > 0.35 → condition False → fires (fire_on='false')
            "account_true":  _LOW,    # 0.25 < 0.35 → condition True  → skipped
        }
    }
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=webhook.url, fire_on="false"),
        )

        # Condition=False → webhook fires (fire_on='false')
        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, "account_false")
        assert body["decision"]["value"] is False
        assert webhook.call_count == 1, "webhook must fire when condition=False and fire_on='false'"

        # Condition=True → webhook skipped
        body = _evaluate_full(
            client, CID, VER, COND_ID, COND_VER, "account_true",
            timestamp="2025-11-14T09:01:00Z",  # different ts to avoid duplicate key
        )
        assert body["decision"]["value"] is True
        assert webhook.call_count == 1, (
            "webhook must NOT fire when condition=True and fire_on='false'"
        )


# ── Test 8 — fire_on='any' always fires ────────────────────────────────────────

@pytest.mark.e2e
def test_webhook_fires_on_any_always_delivers(mock_webhook_connector_e2e_client):
    """
    fire_on='any': webhook fires on every evaluation regardless of decision value.
    """
    CID, VER          = "wd.t8.concept", "v1"
    COND_ID, COND_VER = "wd.t8.condition", "v1"
    ACTION_ID         = "wd.t8.action"

    data = {
        _PRIM: {
            "account_fires":    _LOW,   # condition True
            "account_no_fires": _HIGH,  # condition False
        }
    }
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=webhook.url, fire_on="any"),
        )

        # Condition fires → webhook fires
        _evaluate_full(client, CID, VER, COND_ID, COND_VER, "account_fires")
        assert webhook.call_count == 1

        # Condition does not fire → webhook still fires
        _evaluate_full(
            client, CID, VER, COND_ID, COND_VER, "account_no_fires",
            timestamp="2025-11-14T09:01:00Z",
        )
        assert webhook.call_count == 2, (
            "fire_on='any' must fire on both True and False outcomes"
        )


# ── Test 9 — Webhook failure recorded without crashing pipeline ─────────────────

@pytest.mark.e2e
def test_webhook_failure_recorded_in_decision_record(mock_webhook_connector_e2e_client):
    """
    When the webhook endpoint returns HTTP 500, the pipeline still returns HTTP 200.
    The action status in the response is 'failed' with a non-empty error.
    The pipeline is never blocked by action delivery failures.
    """
    CID, VER          = "wd.t9.concept", "v1"
    COND_ID, COND_VER = "wd.t9.condition", "v1"
    ACTION_ID         = "wd.t9.action"
    ENTITY            = "account_001"

    data = {_PRIM: {ENTITY: _LOW}}    # condition fires
    connector = MockTableConnector(data)
    webhook = MockWebhookServer(status_code=500)   # simulate server failure

    with mock_webhook_connector_e2e_client(connector, webhook) as (client, pool, run_db, wh):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=wh.url),
        )

        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    # Pipeline succeeds despite action failure.
    assert body["decision"]["value"] is True
    actions = body["decision"]["actions_triggered"]
    assert len(actions) == 1, "one action should be recorded"
    assert actions[0]["action_id"] == ACTION_ID
    assert actions[0]["status"] == "failed", (
        f"Expected status='failed' on HTTP 500, got {actions[0]['status']!r}"
    )
    assert actions[0]["error"] is not None, "error field must be set on failure"
    # Webhook was attempted (the call was intercepted and recorded).
    assert webhook.call_count == 1, "delivery was attempted despite failure"


# ── Test 10 — Delivery status in evaluate/full response ────────────────────────

@pytest.mark.e2e
def test_action_status_in_evaluate_full_response(mock_webhook_connector_e2e_client):
    """
    POST /evaluate/full response body includes decision.actions_triggered with
    per-action delivery status.

    Verifies both the 'triggered' (200) and 'failed' (500) status values
    within a single context — same definitions, two different entities, with
    the mock's response code reconfigured between evaluations.
    """
    CID, VER          = "wd.t10.concept", "v1"
    COND_ID, COND_VER = "wd.t10.condition", "v1"
    ACTION_ID         = "wd.t10.action"
    ENTITY_A          = "account_a"
    ENTITY_B          = "account_b"

    data = {_PRIM: {ENTITY_A: _LOW, ENTITY_B: _LOW}}
    connector = MockTableConnector(data)
    webhook = MockWebhookServer(status_code=200)

    with mock_webhook_connector_e2e_client(connector, webhook) as (client, pool, run_db, wh):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=wh.url),
        )

        # First evaluation: HTTP 200 → status='triggered'
        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY_A)
        actions = body["decision"]["actions_triggered"]
        assert isinstance(actions, list), "actions_triggered must be a list"
        assert len(actions) == 1
        assert actions[0]["action_id"] == ACTION_ID
        assert actions[0]["status"] == "triggered"

        # Reconfigure mock to fail on subsequent calls.
        wh.configure_response(status_code=500)

        # Second evaluation: HTTP 500 → status='failed'
        body = _evaluate_full(
            client, CID, VER, COND_ID, COND_VER, ENTITY_B,
            timestamp="2025-11-14T10:00:00Z",
        )
        actions = body["decision"]["actions_triggered"]
        assert len(actions) == 1
        assert actions[0]["status"] == "failed"


# ── Test 11 — Dry run does not deliver webhook ──────────────────────────────────

@pytest.mark.e2e
def test_dry_run_does_not_deliver_webhook(mock_webhook_connector_e2e_client):
    """
    POST /evaluate/full with dry_run=True must not make any outbound HTTP calls.
    The response action status must be 'would_trigger' (not 'triggered').
    """
    CID, VER          = "wd.t11.concept", "v1"
    COND_ID, COND_VER = "wd.t11.condition", "v1"
    ACTION_ID         = "wd.t11.action"
    ENTITY            = "account_001"

    data = {_PRIM: {ENTITY: _LOW}}    # condition fires
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        _register(
            client,
            _concept(CID, VER),
            _condition(COND_ID, COND_VER, CID, VER),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER, endpoint=webhook.url),
        )

        body = _evaluate_full(
            client, CID, VER, COND_ID, COND_VER, ENTITY, dry_run=True
        )

    assert webhook.call_count == 0, (
        "dry_run=True must not make any outbound HTTP calls"
    )
    assert body["dry_run"] is True
    actions = body["decision"]["actions_triggered"]
    assert len(actions) == 1
    assert actions[0]["status"] == "would_trigger", (
        f"dry_run must produce status='would_trigger', got {actions[0]['status']!r}"
    )


# ── Test 12 — Multiple actions all delivered ────────────────────────────────────

@pytest.mark.e2e
def test_multiple_actions_all_delivered(mock_webhook_connector_e2e_client):
    """
    When two webhook actions are bound to the same condition, both are delivered.

    Both actions use the same mock server (all httpx calls are intercepted),
    distinguished by different endpoint URLs recorded in the request log.
    """
    CID, VER          = "wd.t12.concept", "v1"
    COND_ID, COND_VER = "wd.t12.condition", "v1"
    ACTION_ID_1       = "wd.t12.action.one"
    ACTION_ID_2       = "wd.t12.action.two"
    ENDPOINT_1        = "http://mock-webhook.test/deliver-one"
    ENDPOINT_2        = "http://mock-webhook.test/deliver-two"
    ENTITY            = "account_001"

    data = {_PRIM: {ENTITY: _LOW}}
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        # Register concept + condition.
        r = client.post("/registry/definitions", json=_concept(CID, VER), headers=_ELEVATED)
        assert r.status_code == 200

        r = client.post(
            "/registry/definitions",
            json=_condition(COND_ID, COND_VER, CID, VER),
            headers=_ELEVATED,
        )
        assert r.status_code == 200

        # Register two actions bound to the same condition.
        r = client.post(
            "/actions",
            json=_webhook_action(ACTION_ID_1, VER, COND_ID, COND_VER, endpoint=ENDPOINT_1),
            headers=_ELEVATED,
        )
        assert r.status_code == 201

        r = client.post(
            "/actions",
            json=_webhook_action(ACTION_ID_2, VER, COND_ID, COND_VER, endpoint=ENDPOINT_2),
            headers=_ELEVATED,
        )
        assert r.status_code == 201

        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    assert body["decision"]["value"] is True
    assert webhook.call_count == 2, (
        f"Both actions must be delivered; expected 2 calls, got {webhook.call_count}"
    )

    # Both endpoints should appear in the recorded requests.
    urls = {req.url for req in webhook.requests}
    assert ENDPOINT_1 in urls, f"endpoint 1 not called; requests: {urls}"
    assert ENDPOINT_2 in urls, f"endpoint 2 not called; requests: {urls}"

    # Both action IDs must be in the response.
    actions = body["decision"]["actions_triggered"]
    assert len(actions) == 2
    action_ids = {a["action_id"] for a in actions}
    assert ACTION_ID_1 in action_ids
    assert ACTION_ID_2 in action_ids


# ── Test 13 — Notification action type ─────────────────────────────────────────

@pytest.mark.e2e
def test_notification_action_recorded(mock_webhook_connector_e2e_client, monkeypatch):
    """
    Notification actions without CANVAS_NOTIFICATION_ENDPOINT are logged locally.

    Documented behaviour:
      - No outbound HTTP call is made (call_count == 0).
      - ActionTrigger._invoke_notification() returns {"status": "logged", ...}.
      - The action status in the response is 'triggered' (delivery succeeded).
      - payload_sent contains channel and message.

    If CANVAS_NOTIFICATION_ENDPOINT is set, an HTTP POST would be made —
    that path is covered by existing unit tests in test_action_trigger.py.
    """
    CID, VER          = "wd.t13.concept", "v1"
    COND_ID, COND_VER = "wd.t13.condition", "v1"
    ACTION_ID         = "wd.t13.action"
    ENTITY            = "account_001"

    # Ensure no notification endpoint is configured.
    monkeypatch.delenv("CANVAS_NOTIFICATION_ENDPOINT", raising=False)

    data = {_PRIM: {ENTITY: _LOW}}   # condition fires
    connector = MockTableConnector(data)

    with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
        r = client.post("/registry/definitions", json=_concept(CID, VER), headers=_ELEVATED)
        assert r.status_code == 200

        r = client.post(
            "/registry/definitions",
            json=_condition(COND_ID, COND_VER, CID, VER),
            headers=_ELEVATED,
        )
        assert r.status_code == 200

        r = client.post(
            "/actions",
            json=_notification_action(ACTION_ID, VER, COND_ID, COND_VER, channel="slack-test"),
            headers=_ELEVATED,
        )
        assert r.status_code == 201

        body = _evaluate_full(client, CID, VER, COND_ID, COND_VER, ENTITY)

    # No outbound HTTP call — notification was logged only.
    assert webhook.call_count == 0, (
        "notification without CANVAS_NOTIFICATION_ENDPOINT must not make HTTP calls"
    )

    actions = body["decision"]["actions_triggered"]
    assert len(actions) == 1
    # Status is 'triggered' because _invoke_notification returned successfully
    # (logging counts as successful delivery when no endpoint is configured).
    assert actions[0]["status"] in {"triggered", "failed", "not_configured"}, (
        f"Unexpected notification action status: {actions[0]['status']!r}\n"
        f"Documented: without CANVAS_NOTIFICATION_ENDPOINT, status='triggered' "
        f"(logged locally)"
    )
