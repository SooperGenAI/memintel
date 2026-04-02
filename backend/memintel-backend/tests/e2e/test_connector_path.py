"""
tests/e2e/test_connector_path.py
──────────────────────────────────────────────────────────────────────────────
Connector-path e2e tests — real evaluation pipeline with MockTableConnector.

These tests validate that POST /evaluate/full fetches primitive values via the
connector layer (not inline data) and that the full ψ → φ → α pipeline
evaluates correctly end-to-end.

Contrast with test_pipeline_e2e.py workaround W3:
  W3: "evaluate/full has no data field. MockConnector returns None for all
      primitives; missing_data_policy='zero' substitutes 0.0."

These tests RESOLVE W3 by injecting MockTableConnector (via the
mock_connector_e2e_client fixture) which returns deterministic pre-programmed
values per (primitive, entity) pair — no inline data needed.

Design decisions
────────────────
All concepts use z_score_op on a float primitive. z_score_op without a
historical population returns the raw primitive value, so:
  concept_result.value == primitive_value (as fetched by MockTableConnector).

This makes assertions straightforward:
  primitive_value → concept_value → threshold comparison → decision.value.

Test isolation
──────────────
Each test uses unique concept/condition IDs (conn.t1.*, conn.t2.*, …).
The e2e_setup fixture truncates all tables before each test so there are no
cross-test conflicts.

Tests
─────
 1. Basic fetch — connector returns value, threshold fires correctly
 2. Null primitive — connector returns None → null_input policy → null_input reason
 3. Multiple primitives — two primitives fetched, both in input_primitives
 4. Determinism — same entity+timestamp yields identical results across calls
 5. Time series — list primitive → pct_change op → threshold on computed change
 6. Retry on transient error — first call fails, retry succeeds, HTTP 200
 7. Permanent failure — auth error → graceful degradation → null_input
 8. Categorical primitive — equals strategy, matched/unmatched
 9. Timestamp forwarding — as_of timestamp received by connector unchanged
10. History accumulation — z_score strategy works after 3+ stored results
11. REST connector path — async MockAsyncRestConnector via async_connector_registry
"""
from __future__ import annotations

import time
import pytest

from tests.mocks.mock_connector import MockTableConnector
from tests.mocks.mock_rest_connector import MockAsyncRestConnector

# ── Shared helpers ─────────────────────────────────────────────────────────────

_ELEVATED = {"X-Elevated-Key": "contract-test-elevated-key"}
_API      = {"X-Api-Key": "contract-test-api-key"}


def _register(client, concept_body: dict, condition_body: dict, action_body: dict):
    """Register concept, condition, and action via the registry endpoints."""
    r = client.post("/registry/definitions", json=concept_body, headers=_ELEVATED)
    assert r.status_code == 200, f"concept register failed: {r.text}"

    r = client.post("/registry/definitions", json=condition_body, headers=_ELEVATED)
    assert r.status_code == 200, f"condition register failed: {r.text}"

    r = client.post("/actions", json=action_body, headers=_ELEVATED)
    assert r.status_code == 201, f"action register failed: {r.text}"


def _float_concept(cid: str, ver: str, primitive: str, *, op: str = "z_score_op",
                   policy: str = "zero") -> dict:
    """Build a float concept definition body."""
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
            "primitives": {primitive: {"type": "float", "missing_data_policy": policy}},
            "features": {
                "out": {"op": op, "inputs": {"input": primitive}, "params": {}},
            },
        },
    }


def _threshold_condition(cid: str, ver: str, concept_id: str, concept_ver: str,
                         *, direction: str, value: float) -> dict:
    """Build a threshold condition definition body."""
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
                "params": {"direction": direction, "value": value},
            },
        },
    }


def _webhook_action(action_id: str, ver: str, condition_id: str, condition_ver: str) -> dict:
    return {
        "action_id": action_id,
        "version": ver,
        "namespace": "org",
        "config": {"type": "webhook", "endpoint": "https://mock.example.com/connector-test"},
        "trigger": {"fire_on": "true", "condition_id": condition_id, "condition_version": condition_ver},
    }


# ── Test 1: Basic connector fetch ──────────────────────────────────────────────

@pytest.mark.e2e
def test_evaluate_full_fetches_primitive_via_connector(mock_connector_e2e_client, elevated_headers, api_headers):
    """
    Real evaluation with connector data — verifies that /evaluate/full fetches
    primitive values from MockTableConnector (not inline data).

    account_001: 0.25 < 0.35 → fires (True)
    account_002: 0.85 < 0.35 → no fire (False)
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t1.concept", "v1"
    COND_ID, COND_VER = "conn.t1.condition", "v1"
    ACTION_ID = "conn.t1.action"
    TS = "2025-11-14T09:00:00Z"

    data = {PRIM: {"account_001": 0.25, "account_002": 0.85}}
    connector = MockTableConnector(data)

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        _register(
            client,
            _float_concept(CID, VER, PRIM),
            _threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER),
        )

        # account_001: 0.25 < 0.35 → fires
        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": "account_001", "timestamp": TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["result"]["value"] == pytest.approx(0.25), (
            f"concept_value should be 0.25 (fetched via connector), got {body['result']['value']}"
        )
        assert body["decision"]["value"] is True, (
            f"0.25 < 0.35 should fire, got {body['decision']['value']}"
        )

        # account_002: 0.85 < 0.35 → no fire
        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": "account_002", "timestamp": TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["result"]["value"] == pytest.approx(0.85)
        assert body["decision"]["value"] is False

    # Verify the connector was actually called (not using inline data)
    assert connector.fetch_call_count > 0, "MockTableConnector was never called — connector not wired"


# ── Test 2: Null primitive value handling ──────────────────────────────────────

@pytest.mark.e2e
def test_evaluate_full_handles_null_primitive(mock_connector_e2e_client):
    """
    Connector returns None → missing_data_policy='null' → concept_value=None
    → strategy returns reason='null_input'. Never HTTP 500.
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t2.concept", "v1"
    COND_ID, COND_VER = "conn.t2.condition", "v1"
    ACTION_ID = "conn.t2.action"

    # Policy "null" preserves None → concept_value=None → null_input reason.
    data = {PRIM: {"account_null": None}}
    connector = MockTableConnector(data)

    # Build concept with missing_data_policy="null" (not "zero").
    concept_body = {
        "definition_id": CID,
        "version": VER,
        "definition_type": "concept",
        "namespace": "org",
        "body": {
            "concept_id": CID, "version": VER,
            "namespace": "org", "output_type": "float",
            "output_feature": "out",
            "primitives": {PRIM: {"type": "float", "missing_data_policy": "null"}},
            "features": {"out": {"op": "z_score_op", "inputs": {"input": PRIM}, "params": {}}},
        },
    }

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        client.post("/registry/definitions", json=concept_body, headers=_ELEVATED)
        client.post(
            "/registry/definitions",
            json=_threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            headers=_ELEVATED,
        )
        client.post("/actions", json=_webhook_action(ACTION_ID, VER, COND_ID, COND_VER), headers=_ELEVATED)

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": "account_null", "timestamp": "2025-11-14T09:00:00Z",
            },
            headers=_API,
        )
        # Must not be HTTP 500 — graceful null handling
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["decision"]["reason"] == "null_input", (
            f"Expected reason='null_input', got {body['decision']['reason']!r}"
        )
        assert body["decision"]["value"] is False


# ── Test 3: Multiple primitives fetched ────────────────────────────────────────

@pytest.mark.e2e
def test_evaluate_full_fetches_multiple_primitives(mock_connector_e2e_client):
    """
    Two primitives fetched for the same entity. Both appear in the connector's
    call_log and in the decision record's input_primitives field.
    """
    PRIM_RATE = "account.active_user_rate_30d"
    PRIM_DAYS = "account.days_to_renewal"
    CID, VER = "conn.t3.concept", "v1"
    COND_ID, COND_VER = "conn.t3.condition", "v1"
    ACTION_ID = "conn.t3.action"
    ENTITY = "account_multi"
    TS = "2025-11-14T09:00:00Z"

    data = {
        PRIM_RATE: {ENTITY: 0.25},
        PRIM_DAYS: {ENTITY: 30.0},
    }
    connector = MockTableConnector(data)

    # Concept: out = z_score(rate) + z_score(days)
    # 0.25 + 30.0 = 30.25 < 100 → fires
    concept_body = {
        "definition_id": CID, "version": VER,
        "definition_type": "concept", "namespace": "org",
        "body": {
            "concept_id": CID, "version": VER, "namespace": "org",
            "output_type": "float", "output_feature": "combined",
            "primitives": {
                PRIM_RATE: {"type": "float", "missing_data_policy": "zero"},
                PRIM_DAYS: {"type": "float", "missing_data_policy": "zero"},
            },
            "features": {
                "rate_f": {"op": "z_score_op", "inputs": {"input": PRIM_RATE}, "params": {}},
                "days_f": {"op": "z_score_op", "inputs": {"input": PRIM_DAYS}, "params": {}},
                "combined": {"op": "add", "inputs": {"a": "rate_f", "b": "days_f"}, "params": {}},
            },
        },
    }

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        client.post("/registry/definitions", json=concept_body, headers=_ELEVATED)
        client.post(
            "/registry/definitions",
            json=_threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=100.0),
            headers=_ELEVATED,
        )
        client.post("/actions", json=_webhook_action(ACTION_ID, VER, COND_ID, COND_VER), headers=_ELEVATED)

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": ENTITY, "timestamp": TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # 0.25 + 30.0 = 30.25 < 100 → fires
        assert body["decision"]["value"] is True
        assert body["result"]["value"] == pytest.approx(30.25)

    # Both primitives must have been fetched from the connector
    fetched_prims = {log[0] for log in connector.call_log}
    assert PRIM_RATE in fetched_prims, f"{PRIM_RATE} not fetched"
    assert PRIM_DAYS in fetched_prims, f"{PRIM_DAYS} not fetched"


# ── Test 4: Determinism with connector ────────────────────────────────────────

@pytest.mark.e2e
def test_evaluate_full_deterministic_with_connector(mock_connector_e2e_client):
    """
    Same entity + timestamp → identical results across three calls.
    Determinism holds even when values come from the connector.
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t4.concept", "v1"
    COND_ID, COND_VER = "conn.t4.condition", "v1"
    ACTION_ID = "conn.t4.action"
    ENTITY = "account_det"
    TS = "2025-11-14T09:00:00Z"

    data = {PRIM: {ENTITY: 0.25}}
    connector = MockTableConnector(data)

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        _register(
            client,
            _float_concept(CID, VER, PRIM),
            _threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER),
        )

        results = []
        for _ in range(3):
            r = client.post(
                "/evaluate/full",
                json={
                    "concept_id": CID, "concept_version": VER,
                    "condition_id": COND_ID, "condition_version": COND_VER,
                    "entity": ENTITY, "timestamp": TS,
                },
                headers=_API,
            )
            assert r.status_code == 200, r.text
            results.append(r.json())

    values = [res["result"]["value"] for res in results]
    decisions = [res["decision"]["value"] for res in results]
    assert len(set(values)) == 1, f"Non-deterministic concept values: {values}"
    assert len(set(decisions)) == 1, f"Non-deterministic decisions: {decisions}"
    assert values[0] == pytest.approx(0.25)
    assert decisions[0] is True


# ── Test 5: Time series primitive ─────────────────────────────────────────────

@pytest.mark.e2e
def test_evaluate_full_fetches_time_series(mock_connector_e2e_client):
    """
    Connector returns a list (time series) for a primitive. The pct_change op
    computes the relative change between the last two elements. A downward-
    trending series produces a negative pct_change → threshold below 0 fires.

    Series: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pct_change = (0.1 - 0.2) / |0.2| = -0.5
    Threshold below 0 → True (fires).
    """
    PRIM = "user.session_frequency_trend_8w"
    CID, VER = "conn.t5.concept", "v1"
    COND_ID, COND_VER = "conn.t5.condition", "v1"
    ACTION_ID = "conn.t5.action"
    ENTITY = "user_trend"

    series = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    data = {PRIM: {ENTITY: series}}
    connector = MockTableConnector(data)

    # pct_change op converts the time_series to a scalar: (last - prev) / |prev|
    concept_body = {
        "definition_id": CID, "version": VER,
        "definition_type": "concept", "namespace": "org",
        "body": {
            "concept_id": CID, "version": VER, "namespace": "org",
            "output_type": "float", "output_feature": "out",
            "primitives": {PRIM: {"type": "time_series<float>", "missing_data_policy": "null"}},
            "features": {
                "out": {"op": "pct_change", "inputs": {"input": PRIM}, "params": {}},
            },
        },
    }

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        client.post("/registry/definitions", json=concept_body, headers=_ELEVATED)
        client.post(
            "/registry/definitions",
            json=_threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.0),
            headers=_ELEVATED,
        )
        client.post("/actions", json=_webhook_action(ACTION_ID, VER, COND_ID, COND_VER), headers=_ELEVATED)

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": ENTITY, "timestamp": "2025-11-14T09:00:00Z",
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # pct_change = (0.1 - 0.2) / 0.2 = -0.5 < 0 → fires
        assert body["result"]["value"] == pytest.approx(-0.5), (
            f"pct_change of declining series should be -0.5, got {body['result']['value']}"
        )
        assert body["decision"]["value"] is True, "Downward trend should fire threshold below 0"

    assert connector.fetch_call_count > 0, "Connector never called for time series primitive"


# ── Test 6: Connector retry on transient error ─────────────────────────────────

@pytest.mark.e2e
def test_connector_retries_on_transient_error(mock_connector_e2e_client):
    """
    MockTableConnector raises TransientConnectorError on the first call then
    succeeds. DataResolver retries (max_retries=3) and the pipeline completes
    with the correct value. Never HTTP 500.
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t6.concept", "v1"
    COND_ID, COND_VER = "conn.t6.condition", "v1"
    ACTION_ID = "conn.t6.action"
    ENTITY = "account_retry"

    # transient_failures=1: first call raises TransientConnectorError, retry succeeds
    data = {PRIM: {ENTITY: 0.25}}
    connector = MockTableConnector(data, transient_failures=1)

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        _register(
            client,
            _float_concept(CID, VER, PRIM),
            _threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER),
        )

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": ENTITY, "timestamp": "2025-11-14T09:00:00Z",
            },
            headers=_API,
        )
        assert r.status_code == 200, f"Retry should succeed; got {r.status_code}: {r.text}"
        body = r.json()
        assert body["result"]["value"] == pytest.approx(0.25), (
            "After retry, connector should have returned 0.25"
        )
        assert body["decision"]["value"] is True

    # Connector must have been called at least twice (1 failure + 1 success)
    assert connector.fetch_call_count >= 2, (
        f"Expected >= 2 calls (1 failure + 1 retry), got {connector.fetch_call_count}"
    )


# ── Test 7: Connector permanent failure ───────────────────────────────────────

@pytest.mark.e2e
def test_connector_permanent_failure_returns_null_input(mock_connector_e2e_client):
    """
    MockTableConnector raises AuthConnectorError (permanent, never retried).
    DataResolver catches it, returns PrimitiveValue(fetch_error=True, value=None).
    Null propagates through the concept → strategy returns reason='null_input'.
    Never HTTP 500.
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t7.concept", "v1"
    COND_ID, COND_VER = "conn.t7.condition", "v1"
    ACTION_ID = "conn.t7.action"

    # auth_failure=True: every call raises AuthConnectorError
    connector = MockTableConnector(auth_failure=True)

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        _register(
            client,
            _float_concept(CID, VER, PRIM),
            _threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER),
        )

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": "account_auth_fail", "timestamp": "2025-11-14T09:00:00Z",
            },
            headers=_API,
        )
        # Graceful degradation — never HTTP 500
        assert r.status_code == 200, f"Permanent failure should degrade gracefully; got {r.status_code}: {r.text}"
        body = r.json()
        # concept_result.value = None → strategy returns null_input
        assert body["decision"]["reason"] in ("null_input", "fetch_error"), (
            f"Expected null_input or fetch_error reason, got {body['decision']['reason']!r}"
        )
        assert body["decision"]["value"] is False


# ── Test 8: Categorical primitive via connector ────────────────────────────────

@pytest.mark.e2e
def test_evaluate_full_categorical_primitive_via_connector(mock_connector_e2e_client):
    """
    Categorical primitive fetched from connector, evaluated by equals strategy.

    account_failed → "true" → equals("true") → decision.value == "true"
    account_ok     → "false" → equals("true") no match → reason="no_match"
    """
    PRIM = "account.payment_failed_flag"
    LABELS = ["true", "false"]
    CID, VER = "conn.t8.concept", "v1"
    COND_ID, COND_VER = "conn.t8.condition", "v1"
    ACTION_ID = "conn.t8.action"
    TS = "2025-11-14T09:00:00Z"

    data = {PRIM: {"account_failed": "true", "account_ok": "false"}}
    connector = MockTableConnector(data)

    concept_body = {
        "definition_id": CID, "version": VER,
        "definition_type": "concept", "namespace": "org",
        "body": {
            "concept_id": CID, "version": VER, "namespace": "org",
            "output_type": "categorical", "labels": LABELS,
            "output_feature": "flag",
            "primitives": {
                PRIM: {"type": "categorical", "missing_data_policy": "null", "labels": LABELS},
            },
            "features": {
                "flag": {"op": "passthrough", "inputs": {"input": PRIM}, "params": {}},
            },
        },
    }
    condition_body = {
        "definition_id": COND_ID, "version": COND_VER,
        "definition_type": "condition", "namespace": "org",
        "body": {
            "condition_id": COND_ID, "version": COND_VER,
            "namespace": "org", "concept_id": CID, "concept_version": VER,
            "strategy": {"type": "equals", "params": {"value": "true", "labels": LABELS}},
        },
    }

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        client.post("/registry/definitions", json=concept_body, headers=_ELEVATED)
        client.post("/registry/definitions", json=condition_body, headers=_ELEVATED)
        client.post("/actions", json=_webhook_action(ACTION_ID, VER, COND_ID, COND_VER), headers=_ELEVATED)

        # account_failed: returns "true" → equals match
        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": "account_failed", "timestamp": TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"]["value"] == "true", (
            f"Matched 'true' → decision.value should be 'true', got {body['decision']['value']!r}"
        )
        assert body["decision"]["reason"] is None

        # account_ok: returns "false" → no match
        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": "account_ok", "timestamp": TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"]["value"] is None, (
            f"'false' != 'true' → decision.value should be None, got {body['decision']['value']!r}"
        )
        assert body["decision"]["reason"] == "no_match"


# ── Test 9: as_of timestamp forwarded to connector ────────────────────────────

@pytest.mark.e2e
def test_connector_receives_correct_timestamp(mock_connector_e2e_client):
    """
    The as_of timestamp provided in the request body is forwarded unchanged
    to the connector's fetch() call. Inspect MockTableConnector.call_log to
    verify point-in-time fetching is working end-to-end.
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t9.concept", "v1"
    COND_ID, COND_VER = "conn.t9.condition", "v1"
    ACTION_ID = "conn.t9.action"
    ENTITY = "account_ts"
    EXPECTED_TS = "2025-11-14T09:00:00Z"

    data = {PRIM: {ENTITY: 0.25}}
    connector = MockTableConnector(data)

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        _register(
            client,
            _float_concept(CID, VER, PRIM),
            _threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER),
        )

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": ENTITY, "timestamp": EXPECTED_TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text

    # Find fetch calls for our primitive
    relevant = [(prim, eid, ts) for prim, eid, ts in connector.call_log if prim == PRIM]
    assert relevant, f"Connector never fetched {PRIM}"

    timestamps_received = {ts for _, _, ts in relevant}
    assert EXPECTED_TS in timestamps_received, (
        f"Connector should have received timestamp={EXPECTED_TS!r}, "
        f"got {timestamps_received}"
    )


# ── Test 10: History accumulates via connector ────────────────────────────────

@pytest.mark.e2e
def test_history_accumulates_via_connector(mock_connector_e2e_client):
    """
    Execute /evaluate/full 4 times for the same entity with different connector
    values. Each execution stores a concept_result. By the 4th execution the
    z_score strategy has >= 3 history points and evaluates normally
    (reason=None, history_count >= 3).

    Values 0.1, 0.2, 0.3, 0.4 ensure non-zero variance so z_score can compute.
    """
    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t10.concept", "v1"
    COND_ID, COND_VER = "conn.t10.condition", "v1"
    ACTION_ID = "conn.t10.action"
    ENTITY = "account_hist"

    # Use different timestamps per call so each is a distinct deterministic result.
    timestamps = [
        "2025-11-10T09:00:00Z",
        "2025-11-11T09:00:00Z",
        "2025-11-12T09:00:00Z",
        "2025-11-13T09:00:00Z",
    ]
    values_sequence = [0.1, 0.2, 0.3, 0.4]

    data = {PRIM: {ENTITY: values_sequence[0]}}
    connector = MockTableConnector(data)

    z_score_concept = {
        "definition_id": CID, "version": VER,
        "definition_type": "concept", "namespace": "org",
        "body": {
            "concept_id": CID, "version": VER, "namespace": "org",
            "output_type": "float", "output_feature": "out",
            "primitives": {PRIM: {"type": "float", "missing_data_policy": "zero"}},
            "features": {"out": {"op": "z_score_op", "inputs": {"input": PRIM}, "params": {}}},
        },
    }
    z_score_condition = {
        "definition_id": COND_ID, "version": COND_VER,
        "definition_type": "condition", "namespace": "org",
        "body": {
            "condition_id": COND_ID, "version": COND_VER,
            "namespace": "org", "concept_id": CID, "concept_version": VER,
            "strategy": {"type": "z_score", "params": {"threshold": 2.0, "direction": "above"}},
        },
    }

    with mock_connector_e2e_client(connector) as (client, pool, run_db):
        client.post("/registry/definitions", json=z_score_concept, headers=_ELEVATED)
        client.post("/registry/definitions", json=z_score_condition, headers=_ELEVATED)
        client.post("/actions", json=_webhook_action(ACTION_ID, VER, COND_ID, COND_VER), headers=_ELEVATED)

        last_body = None
        for i, (ts, val) in enumerate(zip(timestamps, values_sequence)):
            # Update connector value for this call
            connector._data[PRIM][ENTITY] = val

            r = client.post(
                "/evaluate/full",
                json={
                    "concept_id": CID, "concept_version": VER,
                    "condition_id": COND_ID, "condition_version": COND_VER,
                    "entity": ENTITY, "timestamp": ts,
                },
                headers=_API,
            )
            assert r.status_code == 200, f"Call {i+1} failed: {r.text}"
            last_body = r.json()

    # 4th execution: history minimum is met — reason must not be "insufficient_history"
    # Note: the z_score strategy does NOT populate history_count on the normal evaluation
    # path (only on "insufficient_history" and "zero_variance" early exits).
    assert last_body is not None
    reason = last_body["decision"].get("reason")
    assert reason != "insufficient_history", (
        "After 4 executions the z_score strategy should have enough history, "
        f"but got reason={reason!r}"
    )
    # Confirm a real evaluation occurred (not blocked by a history gate)
    assert reason is None or reason in ("zero_variance",), (
        f"Unexpected reason after history accumulated: {reason!r}"
    )


# ── Test 11: REST connector path (async connector) ────────────────────────────

@pytest.mark.e2e
def test_rest_connector_fetches_primitive(mock_connector_e2e_client):
    """
    MockAsyncRestConnector is injected into the async_connector_registry path.
    DataResolver.afetch() routes the primitive through the async connector
    (via primitive_sources mapping) and returns the correct PrimitiveValue.

    This exercises the production code path used by PostgresConnector and
    RestConnector — the async fetch branch in DataResolver.afetch().
    """
    from app.models.config import PrimitiveSourceConfig

    PRIM = "account.active_user_rate_30d"
    CID, VER = "conn.t11.concept", "v1"
    COND_ID, COND_VER = "conn.t11.condition", "v1"
    ACTION_ID = "conn.t11.action"
    ENTITY = "account_rest"
    TS = "2025-11-14T09:00:00Z"

    data = {PRIM: {ENTITY: 0.25}}
    async_connector = MockAsyncRestConnector(data, connector_name="rest_mock")

    # primitive_sources tells DataResolver.afetch() to use "rest_mock" for PRIM
    primitive_sources = {
        PRIM: PrimitiveSourceConfig(connector="rest_mock", query="/api/v1/{entity_id}"),
    }

    with mock_connector_e2e_client(
        async_connector=async_connector,
        primitive_sources=primitive_sources,
    ) as (client, pool, run_db):
        _register(
            client,
            _float_concept(CID, VER, PRIM),
            _threshold_condition(COND_ID, COND_VER, CID, VER, direction="below", value=0.35),
            _webhook_action(ACTION_ID, VER, COND_ID, COND_VER),
        )

        r = client.post(
            "/evaluate/full",
            json={
                "concept_id": CID, "concept_version": VER,
                "condition_id": COND_ID, "condition_version": COND_VER,
                "entity": ENTITY, "timestamp": TS,
            },
            headers=_API,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["result"]["value"] == pytest.approx(0.25), (
            f"REST connector should have returned 0.25, got {body['result']['value']}"
        )
        assert body["decision"]["value"] is True

    # Verify the async connector was called (not the sync fallback)
    assert async_connector.fetch_call_count > 0, (
        "MockAsyncRestConnector.fetch() was never called — async path not exercised"
    )
