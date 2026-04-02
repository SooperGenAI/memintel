"""
tests/contract/test_api_contracts.py
──────────────────────────────────────────────────────────────────────────────
API contract tests — 30 tests across 6 groups.

Every test uses a real FastAPI app with a real asyncpg pool wired to the
memintel_test database.  Tests are independent: each gets a fresh TestClient
and a truncated (empty) database.

Groups
──────
  GROUP 1 — OpenAPI schema completeness (3 tests)
  GROUP 2 — Request validation contracts  (6 tests)
  GROUP 3 — Response shape contracts     (12 tests)
  GROUP 4 — Error response consistency    (4 tests)
  GROUP 5 — Pagination contracts          (2 tests)
  GROUP 6 — Determinism contracts         (3 tests)

Bugs found
──────────
  BUG-C1: GET /tasks has no authentication — any caller can enumerate all
           tasks without an API key or elevated key.  Documented in
           test_401_gap_get_tasks_no_auth_returns_200.
  BUG-C2: definition_type query parameter on GET /registry/definitions is a
           plain string (not a validated enum) — invalid values return 200
           with an empty list rather than 422.  Documented in
           test_invalid_definition_type_not_validated.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock

if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import pytest

pytestmark = pytest.mark.contract


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — OpenAPI schema completeness
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenAPISchema:
    """Tests 1-3: the generated OpenAPI schema is complete and correct."""

    def test_openapi_schema_loads_without_error(self, app_client):
        """Test 1: GET /openapi.json returns 200 with a valid OpenAPI schema."""
        client, _ = app_client
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "info" in schema
        assert schema["info"]["title"] in ("Memintel Backend API", "FastAPI")

    def test_all_routes_appear_in_openapi_schema(self, app_client):
        """Test 2: every major route group appears in the OpenAPI paths."""
        client, _ = app_client
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]

        expected_prefixes = [
            "/evaluate/full",
            "/execute",
            "/tasks",
            "/registry/definitions",
            "/actions",
            "/jobs",
            "/context",
            "/guardrails",
            "/feedback",
            "/conditions",
            "/decisions",
        ]
        for prefix in expected_prefixes:
            matched = any(p == prefix or p.startswith(prefix + "/") or p.startswith(prefix + "{") for p in paths)
            assert matched, f"No OpenAPI path matching '{prefix}' found in schema"

    def test_no_empty_response_schemas_in_openapi(self, app_client):
        """Test 3: POST /execute/static has a defined response schema (bug D5 fix).

        Before the fix, POST /execute/static had response_model=None which caused
        FastAPI to emit an empty schema object {}.  The fix sets
        response_model=DecisionValue so the schema is populated.
        """
        client, _ = app_client
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]

        # POST /execute/static must exist in the schema
        assert "/execute/static" in paths, "POST /execute/static not found in OpenAPI paths"
        post_op = paths["/execute/static"].get("post", {})
        assert post_op, "POST operation missing for /execute/static"

        # Must have a 200 response with content
        responses = post_op.get("responses", {})
        assert "200" in responses, "POST /execute/static has no 200 response in schema"
        resp_200 = responses["200"]
        # FastAPI emits content.application/json.schema when response_model is set
        content = resp_200.get("content", {})
        assert content, (
            "POST /execute/static 200 response has no content schema — "
            "response_model may still be None (bug D5)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Request validation contracts
# ══════════════════════════════════════════════════════════════════════════════

class TestRequestValidation:
    """Tests 4-9: FastAPI/Pydantic validation rejects invalid requests with 422."""

    def test_missing_concept_id_returns_422(self, app_client, api_key_headers):
        """Test 4: POST /evaluate/full without concept_id → 422."""
        client, _ = app_client
        r = client.post("/evaluate/full", json={
            "concept_version": "1.0",
            "condition_id": "cond.example",
            "condition_version": "1.0",
            "entity": "e1",
        }, headers=api_key_headers)
        assert r.status_code == 422

    def test_concept_id_too_long_returns_422(self, app_client, api_key_headers):
        """Test 5: POST /evaluate/full with 256-char concept_id → 422 (max_length=255)."""
        client, _ = app_client
        r = client.post("/evaluate/full", json={
            "concept_id": "x" * 256,
            "concept_version": "1.0",
            "condition_id": "cond.example",
            "condition_version": "1.0",
            "entity": "e1",
        }, headers=api_key_headers)
        assert r.status_code == 422

    def test_missing_intent_returns_422(self, app_client):
        """Test 6: POST /tasks without intent field → 422."""
        client, _ = app_client
        r = client.post("/tasks", json={"dry_run": False})
        assert r.status_code == 422

    def test_wrong_feedback_value_returns_422(self, app_client, api_key_headers):
        """Test 7: POST /feedback/decision with invalid feedback enum → 422."""
        client, _ = app_client
        r = client.post("/feedback/decision", json={
            "condition_id": "cond.example",
            "condition_version": "1.0",
            "entity": "e1",
            "timestamp": "2026-01-01T00:00:00Z",
            "feedback": "definitely_wrong_value",
        }, headers=api_key_headers)
        assert r.status_code == 422

    def test_invalid_task_status_enum_returns_422(self, app_client, api_key_headers):
        """Test 8: GET /tasks?status=not_a_real_status → 422."""
        client, _ = app_client
        r = client.get("/tasks?status=not_a_real_status", headers=api_key_headers)
        assert r.status_code == 422

    def test_invalid_definition_type_returns_422(self, app_client, api_key_headers):
        """Test 9: GET /registry/definitions?definition_type=bogus → 422 (BUG-C2 fix).

        After FIX 2, definition_type is a validated Literal enum.
        Invalid values now return 422 instead of silently returning an empty list.
        """
        client, _ = app_client
        r = client.get("/registry/definitions?definition_type=totally_bogus_type",
                       headers=api_key_headers)
        assert r.status_code == 422

    def test_missing_required_query_param_returns_422(self, app_client):
        """Test 9b: GET /actions without required 'namespace' query param → 422."""
        client, _ = app_client
        r = client.get("/actions")
        assert r.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Response shape contracts
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseShapes:
    """Tests 10-21: response bodies conform to documented shapes."""

    # ── Context ───────────────────────────────────────────────────────────────

    def test_get_active_context_returns_404_when_none(self, app_client):
        """Test 10: GET /context with no context created → 404 with error shape."""
        client, _ = app_client
        r = client.get("/context")
        assert r.status_code == 404
        body = r.json()
        # The context service raises NotFoundError (MemintelError subclass)
        # which is handled by memintel_error_handler → {error: {type, message}}
        assert "error" in body, f"Expected error shape, got: {body}"
        assert "type" in body["error"]
        assert "message" in body["error"]

    def test_post_context_returns_201_and_context_shape(self, app_client):
        """Test 11: POST /context → 201 ApplicationContext shape."""
        client, _ = app_client
        r = client.post("/context", json={
            "domain": {
                "description": "contract-test domain",
                "entities": [],
                "decisions": [],
            }
        })
        assert r.status_code == 201
        body = r.json()
        assert "context_id" in body
        assert "version" in body
        assert "domain" in body
        assert body["is_active"] is True

    # ── Guardrails ────────────────────────────────────────────────────────────

    def test_get_active_guardrails_returns_404_when_none(self, app_client):
        """Test 12: GET /guardrails with no API version posted → 404."""
        client, _ = app_client
        r = client.get("/guardrails")
        assert r.status_code == 404
        body = r.json()
        assert "error" in body

    def test_post_guardrails_returns_201_and_version_shape(self, app_client, elevated_headers):
        """Test 13: POST /guardrails → 201 GuardrailsVersion shape."""
        client, _ = app_client
        r = client.post("/guardrails", json={
            "guardrails": {
                "strategy_registry": ["threshold", "percentile"],
                "type_strategy_map": {"float": ["threshold", "percentile"]},
                "parameter_priors": {},
                "bias_rules": {},
                "global_preferred_strategy": "threshold",
                "global_default_strategy": "threshold",
            },
            "change_note": "contract test initial",
        }, headers=elevated_headers)
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
        body = r.json()
        assert "guardrails_id" in body
        assert "version" in body
        assert "guardrails" in body
        assert body["is_active"] is True
        assert body["source"] == "api"

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def test_list_tasks_returns_task_list_shape(self, app_client, api_key_headers):
        """Test 14: GET /tasks → 200 TaskList shape (items, has_more, next_cursor)."""
        client, _ = app_client
        r = client.get("/tasks", headers=api_key_headers)
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert isinstance(body["items"], list)
        assert "has_more" in body
        assert "next_cursor" in body

    def test_get_nonexistent_task_returns_404(self, app_client, api_key_headers):
        """Test 15: GET /tasks/{id} for unknown id → 404."""
        client, _ = app_client
        r = client.get("/tasks/nonexistent-task-id", headers=api_key_headers)
        assert r.status_code == 404
        body = r.json()
        assert "error" in body

    # ── Registry ──────────────────────────────────────────────────────────────

    def test_list_registry_definitions_returns_search_result_shape(self, app_client, api_key_headers):
        """Test 16: GET /registry/definitions → 200 SearchResult shape."""
        client, _ = app_client
        r = client.get("/registry/definitions?namespace=org", headers=api_key_headers)
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert isinstance(body["items"], list)
        assert "has_more" in body
        assert "next_cursor" in body

    def test_register_definition_returns_definition_response(self, app_client, elevated_headers):
        """Test 17: POST /registry/definitions → 200 DefinitionResponse shape.

        The route has status_code=200 (not 201) because it can also return
        an existing matching definition (idempotent registration).
        """
        client, _ = app_client
        r = client.post("/registry/definitions", json={
            "definition_id": "org.test_concept",
            "version": "1.0",
            "definition_type": "concept",
            "namespace": "org",
            "body": {
                "concept_id": "org.test_concept",
                "version": "1.0",
                "namespace": "org",
                "output_type": "float",
                "primitives": {
                    "revenue": {"type": "float", "missing_data_policy": "zero"}
                },
                "features": {
                    "score": {"op": "normalize", "inputs": {"input": "revenue"}, "params": {}}
                },
                "output_feature": "score",
            },
        }, headers=elevated_headers)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert "definition_id" in body
        assert "version" in body

    # ── Actions ───────────────────────────────────────────────────────────────

    def test_list_actions_returns_action_list_shape(self, app_client):
        """Test 18: GET /actions?namespace=org → 200 ActionListResponse shape."""
        client, _ = app_client
        r = client.get("/actions?namespace=org")
        assert r.status_code == 200
        body = r.json()
        assert "actions" in body
        assert isinstance(body["actions"], list)
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_post_action_returns_201(self, app_client, elevated_headers):
        """Test 19: POST /actions → 201 DefinitionResponse shape."""
        client, _ = app_client
        r = client.post("/actions", json={
            "action_id": "notify_contract",
            "version": "1.0",
            "namespace": "org",
            "config": {
                "type": "webhook",
                "endpoint": "https://example.com/hook",
            },
            "trigger": {
                "fire_on": "true",
                "condition_id": "cond.example",
                "condition_version": "1.0",
            },
        }, headers=elevated_headers)
        assert r.status_code == 201
        body = r.json()
        assert "definition_id" in body

    # ── Feedback ──────────────────────────────────────────────────────────────

    def test_post_feedback_returns_404_for_unknown_decision(self, app_client, api_key_headers):
        """Test 20: POST /feedback/decision for a non-existent decision → 404.

        The feedback service requires a matching decision record. With an empty
        database the request always returns 404.
        """
        client, _ = app_client
        r = client.post("/feedback/decision", json={
            "condition_id": "cond.example",
            "condition_version": "1.0",
            "entity": "e1",
            "timestamp": "2026-01-01T00:00:00Z",
            "feedback": "false_positive",
        }, headers=api_key_headers)
        assert r.status_code == 404
        body = r.json()
        assert "error" in body

    # ── Calibration ───────────────────────────────────────────────────────────

    def test_post_calibrate_returns_calibration_result_shape(self, app_client):
        """Test 21: POST /conditions/calibrate → 200 CalibrationResult (always 200).

        The calibrate endpoint never returns 4xx for missing feedback — it returns
        CalibrationResult with status='no_recommendation'.
        """
        client, _ = app_client
        # First register the condition definition so calibrate can find it
        r_reg = client.post("/registry/definitions", json={
            "definition_id": "cond.calibrate_test",
            "version": "1.0",
            "definition_type": "condition",
            "namespace": "org",
            "body": {
                "condition_id": "cond.calibrate_test",
                "version": "1.0",
                "namespace": "org",
                "concept_id": "org.test_concept",
                "concept_version": "1.0",
                "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.5}},
            },
        }, headers={"X-Elevated-Key": ELEVATED_KEY})
        assert r_reg.status_code == 200, f"Expected 200, got {r_reg.status_code}: {r_reg.text}"

        r = client.post("/conditions/calibrate", json={
            "condition_id": "cond.calibrate_test",
            "condition_version": "1.0",
        }, headers={"X-Api-Key": API_KEY})
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert body["status"] in ("recommendation_available", "no_recommendation")
        assert "current_params" in body


# ── import the constant for inlined assertions ─────────────────────────────────
from tests.contract.conftest import API_KEY, ELEVATED_KEY


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Error response shape consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorShapes:
    """Tests 22-25: all error responses use the canonical ErrorResponse shape."""

    def test_404_error_has_canonical_shape(self, app_client, api_key_headers):
        """Test 22: any 404 response → {error: {type, message}} shape."""
        client, _ = app_client
        r = client.get("/tasks/no-such-task-id", headers=api_key_headers)
        assert r.status_code == 404
        body = r.json()
        assert "error" in body, f"Expected 'error' key in 404 body, got: {body}"
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert isinstance(body["error"]["message"], str)
        assert len(body["error"]["message"]) > 0

    def test_get_tasks_without_api_key_returns_401(self, app_client):
        """Test 23: GET /tasks without X-Api-Key header → 401 (BUG-C1 fix).

        FIX 1: GET /tasks now requires the X-Api-Key header when
        app.state.api_key is configured. Missing or wrong key → HTTP 401
        with {error: {type: auth_error}} body.
        """
        client, _ = app_client
        # No X-Api-Key header
        r = client.get("/tasks")
        assert r.status_code == 401, (
            f"Expected 401 after BUG-C1 fix, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_403_error_has_canonical_shape(self, app_client):
        """Test 24: missing elevated key on guarded route → 403 {error: {type=auth_error}}."""
        client, _ = app_client
        r = client.post("/registry/definitions", json={
            "definition_id": "org.x",
            "version": "1.0",
            "definition_type": "concept",
            "namespace": "org",
            "body": {},
        })
        assert r.status_code == 403
        body = r.json()
        assert "error" in body, f"Expected 'error' key in 403 body, got: {body}"
        assert body["error"]["type"] == "auth_error"
        assert "message" in body["error"]

    def test_422_uses_fastapi_detail_format(self, app_client, api_key_headers):
        """Test 25: request validation failure → 422 with FastAPI 'detail' array.

        FastAPI's built-in RequestValidationError handler returns:
          {"detail": [{"loc": [...], "msg": "...", "type": "..."}]}
        This is distinct from the Memintel ErrorResponse format used for 4xx/5xx.
        """
        client, _ = app_client
        r = client.post("/evaluate/full", json={"this_key": "is_not_valid"},
                        headers=api_key_headers)
        assert r.status_code == 422
        body = r.json()
        assert "detail" in body, f"Expected 'detail' key in 422 body, got: {body}"
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0
        first = body["detail"][0]
        assert "msg" in first
        assert "type" in first


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — Pagination contracts
# ══════════════════════════════════════════════════════════════════════════════

class TestPagination:
    """Tests 26-27: pagination metadata is correct and cursor-based."""

    def test_list_tasks_cursor_pagination_fields(self, app_client, api_key_headers):
        """Test 26: GET /tasks supports cursor-based pagination.

        The spec describes cursor (task_id of last item), not offset.
        With an empty DB: has_more=False and next_cursor=None.
        """
        client, _ = app_client
        r = client.get("/tasks?limit=5", headers=api_key_headers)
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert "has_more" in body
        assert "next_cursor" in body
        # Empty DB → no more pages
        assert body["has_more"] is False
        assert body["next_cursor"] is None

    def test_list_actions_total_reflects_db_count(self, app_client, elevated_headers):
        """Test 27: GET /actions total matches number of registered actions."""
        client, _ = app_client

        # Initially zero actions
        r0 = client.get("/actions?namespace=org")
        assert r0.status_code == 200
        assert r0.json()["total"] == 0

        # Register one action
        r_reg = client.post("/actions", json={
            "action_id": "pagination_test_action",
            "version": "1.0",
            "namespace": "org",
            "config": {
                "type": "webhook",
                "endpoint": "https://example.com/hook",
            },
            "trigger": {
                "fire_on": "true",
                "condition_id": "cond.example",
                "condition_version": "1.0",
            },
        }, headers=elevated_headers)
        assert r_reg.status_code == 201

        # Total should now be 1
        r1 = client.get("/actions?namespace=org")
        assert r1.status_code == 200
        assert r1.json()["total"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — Determinism contracts
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Tests 28-30: same inputs always produce the same outputs."""

    def test_same_inputs_same_output(self, app_client, api_key_headers):
        """Test 28: two identical GET /registry/definitions requests → identical response."""
        client, _ = app_client
        r1 = client.get("/registry/definitions?namespace=org", headers=api_key_headers)
        r2 = client.get("/registry/definitions?namespace=org", headers=api_key_headers)
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()

    def test_evaluate_full_without_timestamp_is_snapshot_mode(self, app_client, api_key_headers):
        """Test 29: POST /evaluate/full without timestamp → snapshot mode.

        Without timestamp the service uses the current time (snapshot mode).
        Two back-to-back calls with an empty DB both return the same 404 shape —
        the error is deterministic for a given (concept, condition, entity) triple
        that does not exist.
        """
        client, _ = app_client
        body = {
            "concept_id": "test.snapshot_concept",
            "concept_version": "1.0",
            "condition_id": "test.snapshot_condition",
            "condition_version": "1.0",
            "entity": "entity_snapshot",
        }
        r1 = client.post("/evaluate/full", json=body, headers=api_key_headers)
        r2 = client.post("/evaluate/full", json=body, headers=api_key_headers)
        # Both calls must return the same HTTP status
        assert r1.status_code == r2.status_code
        # Both calls must return the same error type (concept/condition not found)
        b1, b2 = r1.json(), r2.json()
        assert b1 == b2 or (
            b1.get("error", {}).get("type") == b2.get("error", {}).get("type")
        )

    def test_evaluate_full_with_timestamp_is_deterministic(self, app_client, api_key_headers):
        """Test 30: POST /evaluate/full with the same timestamp → identical responses.

        With a fixed timestamp the pipeline is deterministic: primitive fetches
        are replayed from cache (or concept not found, which is also stable).
        Two calls with identical payloads must return identical responses.
        """
        client, _ = app_client
        body = {
            "concept_id": "test.deterministic_concept",
            "concept_version": "1.0",
            "condition_id": "test.deterministic_condition",
            "condition_version": "1.0",
            "entity": "entity_det_1",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        r1 = client.post("/evaluate/full", json=body, headers=api_key_headers)
        r2 = client.post("/evaluate/full", json=body, headers=api_key_headers)
        assert r1.status_code == r2.status_code
        assert r1.json() == r2.json()


# ══════════════════════════════════════════════════════════════════════════════
# FIX VERIFICATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFixes:
    """Explicit regression tests for BUG-C1, BUG-C2, and BUG-C4 fixes."""

    def test_fix1_get_tasks_without_api_key_returns_401(self, app_client):
        """FIX 1 (BUG-C1): GET /tasks without X-Api-Key → 401 auth_error.

        Before fix: GET /tasks had no auth — returned 200 for any caller.
        After fix:  require_api_key dependency enforces X-Api-Key header.
        When api_key is configured on app.state, absent or wrong key → 401.
        """
        client, _ = app_client
        r = client.get("/tasks")   # no X-Api-Key header
        assert r.status_code == 401, (
            f"Expected 401 from require_api_key, got {r.status_code}: {r.text}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_fix1_get_tasks_with_api_key_returns_200(self, app_client, api_key_headers):
        """FIX 1 (BUG-C1): GET /tasks with correct X-Api-Key → 200.

        The fix must not block legitimate callers who supply the key.
        """
        client, _ = app_client
        r = client.get("/tasks", headers=api_key_headers)
        assert r.status_code == 200

    def test_fix2_invalid_definition_type_returns_422(self, app_client, api_key_headers):
        """FIX 2 (BUG-C2): GET /registry/definitions?definition_type=bad → 422.

        Before fix: definition_type was str — invalid values silently returned
        200 with an empty list.
        After fix: definition_type is a validated Literal enum — invalid values
        return HTTP 422 immediately.
        """
        client, _ = app_client
        r = client.get("/registry/definitions?definition_type=not_a_real_type",
                       headers=api_key_headers)
        assert r.status_code == 422, (
            f"Expected 422 from Literal validation, got {r.status_code}: {r.text}"
        )

    def test_fix2_valid_definition_type_still_returns_200(self, app_client, api_key_headers):
        """FIX 2 (BUG-C2): valid definition_type values still return 200."""
        client, _ = app_client
        for valid_type in ("concept", "condition", "action", "primitive", "feature"):
            r = client.get(f"/registry/definitions?definition_type={valid_type}",
                           headers=api_key_headers)
            assert r.status_code == 200, (
                f"definition_type='{valid_type}' should return 200, got {r.status_code}"
            )

    def test_fix3_invalid_body_with_unloaded_guardrails_returns_422_not_500(
        self, app_client
    ):
        """FIX 3 (BUG-C4): POST /tasks invalid body + None guardrails_store → 422.

        Before fix: get_task_authoring_service called guardrails_store.get_guardrails()
        even when the store was unloaded, raising RuntimeError before FastAPI's body
        validation ran — turning valid 422 validation errors into HTTP 500.

        After fix: is_loaded() guard skips get_guardrails() when store is None/unloaded,
        so dependency resolution completes cleanly and FastAPI validates the body.

        The contract test app sets guardrails_store=None (simulating the unloaded case).
        """
        client, _ = app_client
        # Missing required fields: intent, entity_scope, delivery
        r = client.post("/tasks", json={"dry_run": False})
        assert r.status_code != 500, (
            "POST /tasks with unloaded guardrails must never return 500"
        )
        assert r.status_code == 422
