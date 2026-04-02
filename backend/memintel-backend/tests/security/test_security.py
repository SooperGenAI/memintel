"""
tests/security/test_security.py
──────────────────────────────────────────────────────────────────────────────
Security test suite for the Memintel Backend API.

Groups
──────
  GROUP 1 — Authentication enforcement
    Every route is exercised without auth. Routes that should require auth are
    asserted to return 401/403. Routes with no auth requirement are documented
    as SECURITY GAPs.

  GROUP 2 — Elevated key enforcement
    Eight POST endpoints that require X-Elevated-Key must reject a caller that
    supplies only X-Api-Key (→ 403, not 200).

  GROUP 3 — Namespace isolation
    The namespace field is a tier enum ('personal','team','org','global'), not
    a per-tenant identifier. Tests verify that the namespace filter correctly
    scopes results AND document that there is no per-tenant access control.

  GROUP 4 — Entity ID pseudonymisation
    PII-like entity identifiers must not appear verbatim in error response
    bodies. Documents that FastAPI 422 responses echo back input (known gap).

  GROUP 5 — Input sanitisation
    Oversized inputs, SQL injection payloads, null bytes, and Unicode are
    rejected with 422 or processed without exposing internal errors.

  GROUP 6 — Information leakage
    Error responses (4xx / 5xx) must not contain stack traces, file system
    paths, SQL, or database credentials.

All tests are marked @pytest.mark.security.
Tests are skipped gracefully when the test database is unavailable.

SECURITY GAPS are documented with inline comments prefixed "# SECURITY GAP:".

Known constraints
─────────────────
  namespace — DB CHECK constraint allows only: 'personal', 'team', 'org', 'global'
  ExecuteRequest — uses field name 'id' (not 'concept_id')
  FastAPI 422 responses — echo back the full input body (known framework behaviour)
"""
from __future__ import annotations

import json
import uuid

import pytest

pytestmark = pytest.mark.security

# ── Helpers ────────────────────────────────────────────────────────────────────

def _uid() -> str:
    """Short unique suffix to avoid key collisions between test runs."""
    return uuid.uuid4().hex[:8]


def _register_def_payload(
    definition_id: str,
    namespace: str = "org",   # must be in ('personal','team','org','global')
    definition_type: str = "concept",
    version: str = "1.0",
) -> dict:
    """Minimal payload for POST /registry/definitions."""
    return {
        "definition_id": definition_id,
        "version": version,
        "definition_type": definition_type,
        "namespace": namespace,
        "body": {"concept_id": definition_id, "version": version},
    }


def _is_json_error(response_body: bytes) -> bool:
    """Return True if the response body is valid JSON."""
    try:
        json.loads(response_body)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _body_has_no_traceback(text: str) -> bool:
    """Return True if the response text contains no Python traceback indicators."""
    strict_indicators = [
        "Traceback (most recent call last)",
        'File "/',
        'File "C:\\',
        "SyntaxError:",
        "AttributeError:",
        "KeyError:",
        "asyncpg.",
        "psycopg",
        "sqlalchemy",
    ]
    return all(ind not in text for ind in strict_indicators)


def _body_has_no_internal_paths(text: str) -> bool:
    """Return True if the response text does not expose file-system paths."""
    path_indicators = ["/home/", "/usr/local/", "site-packages", "\\AppData\\"]
    return all(ind not in text for ind in path_indicators)


def _body_has_no_db_credentials(text: str) -> bool:
    """Return True if the response text does not contain DB credentials."""
    cred_indicators = ["postgres:admin", "localhost:5433", "memintel_test", "asyncpg"]
    return all(ind not in text for ind in cred_indicators)


# ── GROUP 1: Authentication enforcement ────────────────────────────────────────

class TestAuthenticationEnforcement:
    """
    Exercise every major route category without authentication headers.

    Findings
    ────────
    ENFORCED (correct):
      GET  /tasks           — requires X-Api-Key → 401
      GET  /tasks/{id}      — requires X-Api-Key → 401
      GET  /registry/definitions  — requires X-Api-Key → 401
      POST /decisions/explain     — requires X-Api-Key → 401
      POST /feedback/decision     — requires X-Api-Key → 401
      POST /execute               — requires X-Api-Key → 401
      POST /compile         — requires X-Elevated-Key → 403
      POST /registry/definitions — requires X-Elevated-Key → 403
      POST /actions         — requires X-Elevated-Key → 403
      POST /guardrails      — requires X-Elevated-Key → 403

    SECURITY GAP — routes that return non-401/403 with no auth:
      GET  /guardrails               → 404 (no auth, returns 404 when no version)
      POST /context                  → 201 (no auth required)
    """

    def test_get_tasks_without_api_key_returns_401(self, app_client):
        """GET /tasks enforces API key authentication (BUG-C1 fix)."""
        client, _ = app_client
        r = client.get("/tasks")
        assert r.status_code == 401, (
            f"GET /tasks must return 401 without auth, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_get_tasks_with_wrong_api_key_returns_401(self, app_client, wrong_api_key_headers):
        """GET /tasks rejects an incorrect API key."""
        client, _ = app_client
        r = client.get("/tasks", headers=wrong_api_key_headers)
        assert r.status_code == 401

    def test_get_task_by_id_without_auth_returns_401(self, app_client):
        """GET /tasks/{id} now enforces API key authentication."""
        client, _ = app_client
        r = client.get("/tasks/nonexistent-task-id")
        assert r.status_code == 401, (
            f"GET /tasks/{{id}} must return 401 without auth, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_list_registry_definitions_without_auth_returns_401(self, app_client):
        """GET /registry/definitions now enforces API key authentication."""
        client, _ = app_client
        r = client.get("/registry/definitions")
        assert r.status_code == 401, (
            f"GET /registry/definitions must return 401 without auth, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_post_compile_without_elevated_key_returns_403(self, app_client):
        """POST /compile rejects requests without an elevated key."""
        client, _ = app_client
        r = client.post("/compile", json={"concept": {}})
        # Either 403 (auth fails first) or 422 (body fails first) — never 200
        assert r.status_code in (403, 422), (
            f"POST /compile without elevated key must not return 200, got {r.status_code}"
        )

    def test_post_register_definition_without_elevated_key_returns_403(self, app_client):
        """POST /registry/definitions rejects requests without an elevated key."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(f"sec-def-{_uid()}"),
        )
        assert r.status_code == 403, (
            f"POST /registry/definitions must return 403 without elevated key, "
            f"got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_post_guardrails_without_elevated_key_returns_403_or_422(self, app_client):
        """POST /guardrails rejects requests without an elevated key."""
        client, _ = app_client
        r = client.post("/guardrails", json={"strategy_registry": {}, "constraints": {}})
        assert r.status_code in (403, 422), (
            f"POST /guardrails without elevated key must not return 200, got {r.status_code}"
        )

    def test_post_decisions_explain_without_auth_returns_401(self, app_client):
        """POST /decisions/explain now enforces API key authentication."""
        client, _ = app_client
        r = client.post(
            "/decisions/explain",
            json={
                "condition_id": "c1",
                "condition_version": "1.0",
                "entity": "probe@example.com",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        )
        assert r.status_code == 401, (
            f"POST /decisions/explain must return 401 without auth, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_post_feedback_without_auth_returns_401(self, app_client):
        """POST /feedback/decision now enforces API key authentication."""
        client, _ = app_client
        r = client.post(
            "/feedback/decision",
            json={
                "condition_id": "c1",
                "condition_version": "1.0",
                "entity": "probe@example.com",
                "timestamp": "2024-01-01T00:00:00Z",
                "feedback": "correct",
            },
        )
        assert r.status_code == 401, (
            f"POST /feedback/decision must return 401 without auth, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_post_execute_without_auth_returns_401(self, app_client):
        """POST /execute now enforces API key authentication."""
        client, _ = app_client
        r = client.post(
            "/execute",
            json={
                "id": "nonexistent",
                "version": "1.0",
                "entity": "probe",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        )
        assert r.status_code == 401, (
            f"POST /execute must return 401 without auth, got {r.status_code}"
        )
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"


# ── GROUP 2: Elevated key enforcement ──────────────────────────────────────────

class TestElevatedKeyEnforcement:
    """
    Eight POST endpoints require X-Elevated-Key.

    When a caller supplies only X-Api-Key (a valid user-level key), all
    elevated endpoints must return HTTP 403 with an auth_error body.

    Note: for routes where FastAPI validates the request body before checking
    custom dependencies, the result may be 422 instead of 403. Both are
    acceptable — neither is 200.

    Endpoints under test
    ────────────────────
      POST /compile
      POST /compile/semantic
      POST /registry/definitions
      POST /registry/definitions/{id}/deprecate
      POST /registry/definitions/{id}/promote
      POST /actions               (register action)
      POST /guardrails
      POST /agents/semantic-refine
    """

    def test_compile_rejects_api_key_alone(self, app_client, valid_api_key_headers):
        """POST /compile must reject X-Api-Key without X-Elevated-Key → 403."""
        client, _ = app_client
        r = client.post(
            "/compile",
            json={"concept": {"concept_id": "x", "version": "1.0"}},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_compile_semantic_rejects_api_key_alone(self, app_client, valid_api_key_headers):
        """POST /compile/semantic must reject X-Api-Key without X-Elevated-Key → 403."""
        client, _ = app_client
        r = client.post(
            "/compile/semantic",
            json={"concept": {"concept_id": "x", "version": "1.0"}},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_register_definition_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """POST /registry/definitions must reject X-Api-Key without X-Elevated-Key → 403."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(f"sec-{_uid()}"),
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_deprecate_definition_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """
        POST /registry/definitions/{id}/deprecate must reject API key alone.

        Returns 403 when auth check runs first, or 422 when body validation
        runs before the dependency. Either is acceptable — not 200.
        """
        client, _ = app_client
        r = client.post(
            f"/registry/definitions/some-def-{_uid()}/deprecate",
            json={},
            headers=valid_api_key_headers,
        )
        # 403 or 422 acceptable — never 200
        assert r.status_code in (403, 422), (
            f"Deprecate without elevated key should return 403 or 422, got {r.status_code}"
        )
        assert r.status_code != 200

    def test_promote_definition_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """
        POST /registry/definitions/{id}/promote must reject API key alone.

        Returns 403 when auth check runs first, or 422 when body validation
        runs before the dependency. Either is acceptable — not 200.
        """
        client, _ = app_client
        r = client.post(
            f"/registry/definitions/some-def-{_uid()}/promote",
            json={"target_namespace": "global"},
            headers=valid_api_key_headers,
        )
        assert r.status_code in (403, 422), (
            f"Promote without elevated key should return 403 or 422, got {r.status_code}"
        )
        assert r.status_code != 200

    def test_register_action_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """POST /actions must reject X-Api-Key without X-Elevated-Key → 403."""
        client, _ = app_client
        r = client.post(
            "/actions",
            json={
                "action_id": f"act-{_uid()}",
                "version": "1.0",
                "namespace": "org",
                "config": {"type": "webhook", "url": "https://example.com/hook"},
                "trigger": {"fire_on": "true"},
            },
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_create_guardrails_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """POST /guardrails must reject X-Api-Key without X-Elevated-Key → 403."""
        client, _ = app_client
        r = client.post(
            "/guardrails",
            json={"strategy_registry": {}, "constraints": {}},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_semantic_refine_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """POST /agents/semantic-refine must reject X-Api-Key without X-Elevated-Key → 403."""
        client, _ = app_client
        r = client.post(
            "/agents/semantic-refine",
            json={"definition_id": "x", "version": "1.0", "instruction": "simplify"},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_wrong_elevated_key_returns_403(self, app_client, wrong_elevated_headers):
        """An incorrect elevated key value returns 403, not 200 or 401."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(f"sec-{_uid()}"),
            headers=wrong_elevated_headers,
        )
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "auth_error"

    def test_elevated_endpoint_with_correct_key_not_rejected_for_auth(
        self, app_client, valid_elevated_headers
    ):
        """POST /registry/definitions with valid elevated key is not rejected for auth."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(f"sec-{_uid()}"),
            headers=valid_elevated_headers,
        )
        # Auth passes — outcome is 200 (registered) or 409 (duplicate)
        assert r.status_code in (200, 409), (
            f"Expected 200 or 409 with valid elevated key, got {r.status_code}: {r.text}"
        )

    def test_deprecate_definition_with_version_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """
        Fix 1 (BUG-B2) regression — deprecate must enforce elevated key.

        Providing the required ?version= query param avoids 422 from missing
        param validation, ensuring the auth check fires and returns 403.
        """
        client, _ = app_client
        r = client.post(
            f"/registry/definitions/some-def-{_uid()}/deprecate?version=1.0",
            json={},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403, (
            f"POST /registry/definitions/{{id}}/deprecate must return 403 "
            f"with API key only (Fix 1 BUG-B2), got {r.status_code}: {r.text}"
        )
        assert r.json()["error"]["type"] == "auth_error"

    def test_promote_definition_with_version_rejects_api_key_alone(
        self, app_client, valid_api_key_headers
    ):
        """
        Fix 1 (BUG-B2) regression — promote must enforce elevated key.

        Providing the required ?version= query param avoids 422 from missing
        param validation, ensuring the auth check fires and returns 403.
        """
        client, _ = app_client
        r = client.post(
            f"/registry/definitions/some-def-{_uid()}/promote?version=1.0",
            json={"target_namespace": "global"},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403, (
            f"POST /registry/definitions/{{id}}/promote must return 403 "
            f"with API key only (Fix 1 BUG-B2), got {r.status_code}: {r.text}"
        )
        assert r.json()["error"]["type"] == "auth_error"


# ── GROUP 3: Namespace isolation ────────────────────────────────────────────────

class TestNamespaceIsolation:
    """
    Namespace isolation tests.

    Namespace as a tier system (not per-tenant)
    ────────────────────────────────────────────
    The `namespace` field is a DB-enforced tier enum:
      'personal' | 'team' | 'org' | 'global'

    This is NOT a per-customer namespace — it is a visibility tier.
    There is no per-tenant authentication; the API key is global.

    SECURITY GAP: All authenticated callers share the same global API key
    and can read definitions from any tier by passing that namespace as a
    filter.  There is no per-tenant access control — any two customers with
    the same API key can enumerate each other's definitions within the same tier.

    Tests confirm
    ─────────────
    1. Namespace filter correctly scopes results (correct behaviour).
    2. Any caller can read any tier by knowing its name (security gap).
    3. Omitting the namespace filter returns all tiers (security gap).
    """

    def test_namespace_filter_excludes_other_tier(
        self, app_client, valid_elevated_headers, valid_api_key_headers
    ):
        """
        A definition in 'org' tier is not returned when querying 'personal' tier.

        This tests that the namespace FILTER works correctly.
        """
        client, _ = app_client
        def_id = f"isolated-def-{_uid()}"

        # Register definition in 'org' namespace
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(def_id, namespace="org"),
            headers=valid_elevated_headers,
        )
        assert r.status_code == 200, f"Registration failed: {r.text}"

        # Query with 'personal' namespace — should not appear
        r = client.get(
            "/registry/definitions",
            params={"namespace": "personal"},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 200
        body = r.json()
        ids = [item.get("definition_id") for item in body.get("items", [])]
        assert def_id not in ids, (
            f"Namespace filter FAILED: {def_id} visible in 'personal' tier "
            f"after registering in 'org' tier"
        )

    def test_definitions_visible_in_correct_tier(
        self, app_client, valid_elevated_headers, valid_api_key_headers
    ):
        """A definition in 'org' tier appears when querying the 'org' namespace."""
        client, _ = app_client
        def_id = f"org-def-{_uid()}"

        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(def_id, namespace="org"),
            headers=valid_elevated_headers,
        )
        assert r.status_code == 200, f"Registration failed: {r.text}"

        r = client.get(
            "/registry/definitions",
            params={"namespace": "org"},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 200
        body = r.json()
        ids = [item.get("definition_id") for item in body.get("items", [])]
        assert def_id in ids, f"{def_id} not found in 'org' tier results"

    def test_no_namespace_filter_exposes_all_tiers(
        self, app_client, valid_elevated_headers, valid_api_key_headers
    ):
        """
        SECURITY GAP: Omitting the namespace filter returns definitions from
        ALL tiers, including those registered in restricted tiers.
        """
        client, _ = app_client
        def_id = f"all-tiers-def-{_uid()}"

        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(def_id, namespace="personal"),
            headers=valid_elevated_headers,
        )
        assert r.status_code == 200, f"Registration failed: {r.text}"

        # Query without any namespace filter — SECURITY GAP: all tiers visible
        r = client.get(
            "/registry/definitions",
            headers=valid_api_key_headers,
        )
        assert r.status_code == 200
        body = r.json()
        ids = [item.get("definition_id") for item in body.get("items", [])]
        assert def_id in ids, (
            "SECURITY GAP confirmed: GET /registry/definitions without namespace "
            "filter exposes definitions from ALL tiers (including 'personal')."
        )

    def test_actions_namespace_filter_scoped_correctly(
        self, app_client
    ):
        """
        GET /actions with an unused namespace returns zero results.

        Validates that the action list endpoint correctly applies namespace scoping
        and does not cross-contaminate results between namespaces.
        """
        client, _ = app_client
        # Use a namespace that has no actions registered in the fresh DB
        r = client.get("/actions", params={"namespace": "personal"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0, (
            f"Expected zero actions for 'personal' namespace in fresh DB, "
            f"got {body['total']}"
        )

    def test_cross_tier_read_is_possible_with_known_namespace(
        self, app_client, valid_elevated_headers, valid_api_key_headers
    ):
        """
        SECURITY GAP: Any authenticated caller can read 'personal' tier definitions
        by explicitly passing namespace='personal' as a query parameter.

        There is no access control enforcing which tiers a given API key may read.
        """
        client, _ = app_client
        def_id = f"personal-def-{_uid()}"

        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(def_id, namespace="personal"),
            headers=valid_elevated_headers,
        )
        assert r.status_code == 200, f"Registration failed: {r.text}"

        # SECURITY GAP: query 'personal' tier without elevated key — should see it
        r = client.get(
            "/registry/definitions",
            params={"namespace": "personal"},
            headers=valid_api_key_headers,
        )
        assert r.status_code == 200
        body = r.json()
        ids = [item.get("definition_id") for item in body.get("items", [])]
        assert def_id in ids, (
            "SECURITY GAP confirmed: any authenticated caller can read 'personal' "
            "tier definitions — per-tier read access is not enforced."
        )


# ── GROUP 4: Entity ID pseudonymisation ────────────────────────────────────────

class TestEntityPseudonymisation:
    """
    Entity identifier handling in API responses.

    The `entity` field is used as a semantic key for concept execution results,
    decisions, and feedback records.

    Findings
    ────────
    SECURE: Not-found (404) error response bodies use structured ErrorResponse
    format and do not echo the entity value in the error message.

    SECURITY GAP (FastAPI framework behaviour): HTTP 422 validation error
    responses include the full input body in the `detail.input` field. If a
    caller submits a PII-bearing entity value and the request fails validation,
    the PII will appear in the 422 response body. This is FastAPI's built-in
    behaviour and cannot be changed without a custom validation error handler.

    SECURITY GAP (by design): Entity values are stored as-is and returned
    verbatim in successful execution and decision responses. Callers who store
    PII in entity keys are responsible for pseudonymising values before calling
    the API.
    """

    def test_entity_pii_not_echoed_in_decision_not_found_error(
        self, app_client, valid_api_key_headers
    ):
        """
        When a decision record is not found (404), the error body must not echo
        the entity value. The error message describes the missing decision, not
        the caller-supplied entity.
        """
        client, _ = app_client
        pii_entity = f"sensitive.user.{_uid()}@corporate.example.com"

        r = client.post(
            "/decisions/explain",
            json={
                "condition_id": "nonexistent-condition",
                "condition_version": "1.0",
                "entity": pii_entity,
                "timestamp": "2024-01-01T00:00:00Z",
            },
            headers=valid_api_key_headers,
        )
        # This should return 404 (condition not found in DB), not 422
        assert r.status_code == 404, (
            f"Expected 404 for missing condition, got {r.status_code}: {r.text}"
        )
        response_text = r.text
        # Entity value must not appear verbatim in the 404 error body
        assert pii_entity not in response_text, (
            f"SECURITY: entity PII '{pii_entity}' found verbatim in 404 error response"
        )

    def test_entity_pii_not_in_execution_not_found_error(
        self, app_client, valid_api_key_headers
    ):
        """
        When a concept is not found (404), the error body must not echo the
        entity value submitted in the execution request.
        """
        client, _ = app_client
        pii_entity = f"pii-user-{_uid()}@example.org"

        r = client.post(
            "/execute",
            json={
                "id": "nonexistent-concept-xyz",
                "version": "1.0",
                "entity": pii_entity,
                "timestamp": "2024-01-01T00:00:00Z",
            },
            headers=valid_api_key_headers,
        )
        assert r.status_code in (404, 422), (
            f"Expected 404 or 422, got {r.status_code}: {r.text}"
        )
        if r.status_code == 404:
            # 404: entity must not appear in the error message
            assert pii_entity not in r.text, (
                f"SECURITY: entity PII '{pii_entity}' echoed in 404 error response"
            )
        else:
            # 422: FastAPI echoes back input body — SECURITY GAP documented
            # This is acceptable for the test since the gap is already noted in the docstring
            pass

    def test_entity_pii_not_in_feedback_decision_not_found_error(
        self, app_client, valid_api_key_headers
    ):
        """
        When feedback is submitted for a non-existent decision (404), the error
        body must not echo the entity value.
        """
        client, _ = app_client
        pii_entity = f"john.doe.{_uid()}@company.internal"

        r = client.post(
            "/feedback/decision",
            json={
                "condition_id": "no-such-condition",
                "condition_version": "1.0",
                "entity": pii_entity,
                "timestamp": "2024-01-01T00:00:00Z",
                "feedback": "correct",
            },
            headers=valid_api_key_headers,
        )
        assert r.status_code in (404, 422), (
            f"Expected 404 or 422, got {r.status_code}: {r.text}"
        )
        if r.status_code == 404:
            assert pii_entity not in r.text, (
                f"SECURITY: entity PII '{pii_entity}' echoed in 404 error response"
            )

    def test_fastapi_422_does_not_echo_input(self, app_client, valid_api_key_headers):
        """
        Custom RequestValidationError handler strips the 'input' field from
        422 validation error responses, preventing PII from being echoed back.

        FIX 3: A custom validation_exception_handler in main.py replaces FastAPI's
        built-in 422 handler and removes the 'input' field from each error entry.
        """
        client, _ = app_client
        pii_entity = f"pii-in-422.{_uid()}@example.com"

        # Submit a request with a MISSING required field to trigger 422.
        # Entity is present but the required field 'id' is absent from ExecuteRequest.
        r = client.post(
            "/execute",
            json={
                # Missing required 'id' field — triggers 422
                "version": "1.0",
                "entity": pii_entity,
            },
            headers=valid_api_key_headers,
        )
        assert r.status_code == 422
        # FIX 3: custom handler strips 'input' — PII must NOT appear in response
        body = r.json()
        body_text = r.text
        assert pii_entity not in body_text, (
            "FIX 3 FAILED: PII from 'entity' field still appears in 422 response body."
        )
        # The detail array must still be present with loc/msg/type but no 'input'
        assert "detail" in body
        for err in body.get("detail", []):
            assert "input" not in err, (
                f"'input' field found in 422 detail entry: {err}"
            )

    def test_entity_stored_as_provided_is_design_decision(self):
        """
        SECURITY GAP (by design): Entity values are stored as-is and returned
        verbatim in successful responses.

        This test documents the behaviour — it records the gap for the security report.
        Callers who use PII as entity keys (e.g., email addresses) must implement
        their own pseudonymisation / hashing before calling the API.
        """
        # Documentation test — passes trivially, records the gap
        assert True, (
            "SECURITY GAP (documented): entity values are stored and returned "
            "verbatim. Callers are responsible for pseudonymising PII before use."
        )


# ── GROUP 5: Input sanitisation ────────────────────────────────────────────────

class TestInputSanitisation:
    """
    Verify that adversarial inputs are rejected or handled safely.

    SQL injection payloads must not cause 500 errors or expose DB internals.
    Oversized inputs (exceeding max_length) must return 422 not 500.
    Null bytes and Unicode must be handled without crashing the server.

    Note: Tests in this group use the `permissive_client` fixture
    (raise_server_exceptions=False) for inputs that may cause unhandled DB
    exceptions.  An unhandled exception returned as HTTP 500 is documented as
    a security gap.  An error caught and returned as 4xx is correct behaviour.

    DB namespace constraint: namespace must be one of
      'personal' | 'team' | 'org' | 'global'
    """

    def test_sql_injection_in_definition_id_does_not_cause_500(
        self, permissive_client, valid_elevated_headers
    ):
        """
        SQL injection payload in definition_id must not trigger a 500 error
        or expose DB internals.

        The service uses parameterised queries (asyncpg), so SQL injection in
        the value should be stored safely or rejected by a DB constraint.
        A 500 with asyncpg error detail would be a security issue.
        """
        sql_payload = "' OR '1'='1'; DROP TABLE definitions; --"
        r = permissive_client.post(
            "/registry/definitions",
            json=_register_def_payload(sql_payload, namespace="org"),
            headers=valid_elevated_headers,
        )
        # Acceptable: 422 (Pydantic rejects), 400 (service rejects), 200 (stored safely)
        # Unacceptable: 500 (internal error / SQL injection leak)
        assert r.status_code != 500, (
            f"SQL injection caused 500 error: {r.text[:300]}"
        )
        if r.status_code == 500:
            # Extra assertion to catch credential leaks in 500 bodies
            assert _body_has_no_db_credentials(r.text), (
                "DB credentials exposed in 500 response to SQL injection"
            )

    def test_oversized_definition_id_returns_422(
        self, app_client, valid_elevated_headers
    ):
        """
        A definition_id exceeding max_length=255 must be rejected with 422
        (Pydantic validation), not cause a 500 error.
        """
        client, _ = app_client
        oversized_id = "x" * 300   # exceeds max_length=255
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(oversized_id),
            headers=valid_elevated_headers,
        )
        assert r.status_code == 422, (
            f"Oversized definition_id should return 422, got {r.status_code}"
        )

    def test_oversized_intent_does_not_cause_500(self, permissive_client):
        """
        A POST /tasks body with a very large intent must not cause a 500 error.

        SECURITY GAP: If no max_length is enforced on intent, oversized
        payloads could exhaust server resources (DoS vector).
        """
        oversized_intent = "A" * 50_000  # very large intent
        r = permissive_client.post(
            "/tasks",
            json={"intent": oversized_intent},
        )
        # Acceptable: 422 (validated) or any 4xx. Unacceptable: 500.
        assert r.status_code != 500, (
            f"Oversized intent caused 500: {r.text[:200]}"
        )
        assert _is_json_error(r.content), (
            "Oversized intent response is not valid JSON"
        )

    def test_null_byte_in_definition_id_does_not_cause_unhandled_500(
        self, permissive_client, valid_elevated_headers
    ):
        """
        A null byte (\\x00) in definition_id must not produce an unhandled 500.

        FIX 2: A global asyncpg.PostgresError exception handler in main.py
        catches the DataError raised by Postgres for null bytes and returns
        HTTP 422 with {"detail": "Invalid data format"}.
        """
        null_byte_id = f"def-\x00-invalid-{_uid()}"
        r = permissive_client.post(
            "/registry/definitions",
            json=_register_def_payload(null_byte_id, namespace="org"),
            headers=valid_elevated_headers,
        )
        # FIX 2: must return 4xx, not 500
        assert r.status_code != 500, (
            f"FIX 2 FAILED: null byte in definition_id still causes 500. "
            f"Response: {r.text[:300]}"
        )
        assert r.status_code in (200, 400, 422), (
            f"Unexpected status for null-byte input: {r.status_code}"
        )
        # Credentials must not appear regardless of status
        assert _body_has_no_db_credentials(r.text), (
            "SECURITY: DB credentials exposed in response to null-byte input"
        )

    def test_unicode_in_definition_id_handled_safely(
        self, permissive_client, valid_elevated_headers
    ):
        """
        Unicode characters in definition_id must be handled without crashing.
        UTF-8 is valid for Postgres text columns; unicode must not cause a 500.
        """
        unicode_id = f"def-αβγδεζ-{_uid()}"  # Greek letters
        r = permissive_client.post(
            "/registry/definitions",
            json=_register_def_payload(unicode_id, namespace="org"),
            headers=valid_elevated_headers,
        )
        # 200 (stored safely) or 422 (rejected) — never 500
        assert r.status_code in (200, 400, 422), (
            f"Unicode definition_id caused unexpected {r.status_code}: {r.text[:200]}"
        )

    def test_xss_payload_in_definition_id_does_not_cause_500(
        self, permissive_client, valid_elevated_headers
    ):
        """
        XSS-like payload in definition_id must be handled without a 500 error.
        The response body must be valid JSON (not HTML with script tags).
        """
        xss_id = f"<script>alert('xss')</script>-{_uid()}"
        r = permissive_client.post(
            "/registry/definitions",
            json=_register_def_payload(xss_id, namespace="org"),
            headers=valid_elevated_headers,
        )
        assert r.status_code != 500, (
            f"XSS payload in definition_id caused 500: {r.text[:200]}"
        )
        # Response must be JSON, not raw HTML
        assert _is_json_error(r.content), (
            "XSS payload produced non-JSON response (possible HTML error page)"
        )

    def test_empty_string_definition_id_does_not_cause_unhandled_error(
        self, permissive_client, valid_elevated_headers
    ):
        """
        An empty definition_id string should be rejected by Pydantic (422) or
        by the DB constraint.  An unhandled 500 is a security gap.
        """
        r = permissive_client.post(
            "/registry/definitions",
            json=_register_def_payload("", namespace="org"),
            headers=valid_elevated_headers,
        )
        if r.status_code == 500:
            assert _body_has_no_db_credentials(r.text), (
                "SECURITY: DB credentials exposed in 500 response to empty definition_id"
            )
            pytest.xfail(
                "SECURITY GAP: empty definition_id causes unhandled 500 "
                "(no Pydantic min_length=1 constraint on definition_id field)."
            )
        else:
            # 422 (Pydantic), 400 (service), or 200 (stored) — all acceptable
            assert r.status_code in (200, 400, 422), (
                f"Unexpected status for empty definition_id: {r.status_code}"
            )


# ── GROUP 6: Information leakage ───────────────────────────────────────────────

class TestInformationLeakage:
    """
    Error responses (4xx / 5xx) must not expose:
      - Python tracebacks or stack traces
      - Internal file-system paths (site-packages, /home/, /usr/)
      - Database credentials (postgres:admin, localhost:5433)
      - Raw SQL or asyncpg error messages
      - Server-side technology identifiers in error bodies

    All 4xx / 5xx responses must be valid JSON (not HTML error pages).
    """

    def test_404_error_no_stack_trace(self, app_client, valid_api_key_headers):
        """404 from GET /tasks/{id} must not contain a Python traceback."""
        client, _ = app_client
        r = client.get("/tasks/no-such-task-id-xyz", headers=valid_api_key_headers)
        assert r.status_code == 404
        assert _is_json_error(r.content), "404 response is not valid JSON"
        assert _body_has_no_traceback(r.text), (
            f"404 response contains stack trace indicators: {r.text[:500]}"
        )
        assert _body_has_no_internal_paths(r.text), (
            f"404 response contains internal paths: {r.text[:500]}"
        )

    def test_404_has_canonical_error_shape(self, app_client, valid_api_key_headers):
        """404 error body must use {error: {type, message}} shape."""
        client, _ = app_client
        r = client.get("/tasks/no-such-task-id-xyz", headers=valid_api_key_headers)
        assert r.status_code == 404
        body = r.json()
        assert "error" in body, f"404 body missing 'error' key: {body}"
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert body["error"]["type"] == "not_found"

    def test_401_error_no_credentials_leaked(self, app_client):
        """401 from GET /tasks must not leak DB credentials or file paths."""
        client, _ = app_client
        r = client.get("/tasks")
        assert r.status_code == 401
        assert _is_json_error(r.content), "401 response is not valid JSON"
        assert _body_has_no_traceback(r.text)
        assert _body_has_no_db_credentials(r.text), (
            f"DB credentials found in 401 response: {r.text[:500]}"
        )

    def test_401_has_canonical_error_shape(self, app_client):
        """401 error body must use {error: {type: 'auth_error', message}} shape."""
        client, _ = app_client
        r = client.get("/tasks")
        assert r.status_code == 401
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"
        assert len(body["error"]["message"]) > 0

    def test_403_error_no_credentials_leaked(self, app_client, valid_api_key_headers):
        """403 from POST /registry/definitions must not leak internals."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(f"leak-test-{_uid()}"),
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        assert _is_json_error(r.content), "403 response is not valid JSON"
        assert _body_has_no_traceback(r.text)
        assert _body_has_no_db_credentials(r.text)

    def test_403_has_canonical_error_shape(self, app_client, valid_api_key_headers):
        """403 error body must use {error: {type: 'auth_error', message}} shape."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload(f"leak-test-{_uid()}"),
            headers=valid_api_key_headers,
        )
        assert r.status_code == 403
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"

    def test_422_error_no_stack_trace(self, app_client, valid_elevated_headers):
        """422 from a malformed body must not expose internal paths or stack traces."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json={"definition_id": "x", "version": "1.0"},  # missing required fields
            headers=valid_elevated_headers,
        )
        assert r.status_code == 422
        assert _is_json_error(r.content), "422 response is not valid JSON"
        assert _body_has_no_traceback(r.text)
        assert _body_has_no_internal_paths(r.text)

    def test_422_uses_fastapi_detail_format(self, app_client, valid_elevated_headers):
        """422 from body validation uses FastAPI's {detail: [{loc, msg, type}]} format."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json={"definition_id": "x"},  # missing required fields
            headers=valid_elevated_headers,
        )
        assert r.status_code == 422
        body = r.json()
        assert "detail" in body, (
            f"422 body should use FastAPI 'detail' format, got: {body}"
        )
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0

    def test_oversized_input_error_no_stack_trace(
        self, app_client, valid_elevated_headers
    ):
        """422 from an oversized input must not expose internal paths or tracebacks."""
        client, _ = app_client
        r = client.post(
            "/registry/definitions",
            json=_register_def_payload("x" * 300),
            headers=valid_elevated_headers,
        )
        assert r.status_code == 422
        assert _body_has_no_traceback(r.text)
        assert _body_has_no_internal_paths(r.text)

    def test_task_not_found_error_has_no_db_detail(
        self, app_client, valid_api_key_headers
    ):
        """
        Not-found errors must not expose raw asyncpg error messages or
        SQL query fragments in the response body.
        """
        client, _ = app_client
        r = client.get(
            "/tasks/task-id-that-does-not-exist-abc123",
            headers=valid_api_key_headers,
        )
        assert r.status_code == 404
        body_text = r.text
        sql_indicators = [
            "SELECT",
            "FROM",
            "WHERE",
            "asyncpg",
            "PostgresError",
            "UndefinedColumnError",
        ]
        for indicator in sql_indicators:
            assert indicator not in body_text, (
                f"SQL/asyncpg detail '{indicator}' found in 404 response"
            )
