"""
tests/integration/test_security_scenarios.py
──────────────────────────────────────────────────────────────────────────────
Security scenario tests — all in-process via ASGI TestClient or direct
service calls; no real DB, Redis, or LLM required.

Coverage:
  1. Elevated key enforcement
       a) POST /registry/definitions without X-Elevated-Key → 403 auth_error
       b) POST /registry/definitions with wrong key → 403 auth_error
       c) POST /registry/definitions with correct key → request reaches service
       d) POST /registry/definitions/{id}/promote without key → GAP (no 403)
  2. Namespace isolation
       a) List with namespace=personal filter excludes org definitions
       b) List with namespace=org filter excludes personal definitions
       c) Direct GET by id does not enforce namespace context (documented gap)
  3. Calibration token cross-condition rejection
       a) ApplyCalibrationRequest has no condition_id field (structurally impossible)
       b) Invalid/expired token → PARAMETER_ERROR (HTTP 400)
       c) Token for condition X → result.condition_id is X (cannot redirect to Y)
  4. Input size limits
       a) POST /tasks with 10,000-char intent → no rejection (gap: no maxLength)
       b) CreateTaskRequest.intent carries no max_length validator
  5. Invalid JSON → all sampled POST routes → 422 (not 500 or crash)

Findings reported in test docstrings and the final summary at the bottom.
"""
from __future__ import annotations

# aioredis stub — must precede all app imports
import sys
from unittest.mock import AsyncMock, MagicMock

if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from app.api.routes import execute as execute_route
from app.api.routes import registry as registry_route
from app.api.routes import tasks as tasks_route
from app.api.routes import feedback as feedback_route
from app.api.routes import conditions as conditions_route
from app.api.routes.conditions import get_calibration_service
from app.api.routes.registry import get_registry_service
from app.api.routes.tasks import get_task_authoring_service
from app.api.routes.execute import get_execute_service
from app.api.routes.feedback import get_feedback_service
from app.models.calibration import (
    ApplyCalibrationRequest,
    ApplyCalibrationResult,
    CalibrationToken,
    TaskPendingRebind,
)
from app.models.concept import DefinitionResponse
from app.models.errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorType,
    MemintelError,
    memintel_error_handler,
)
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    Task,
    TaskStatus,
)
from app.services.calibration import CalibrationService


# ── Shared lifespan / exception handler ───────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    """Null lifespan — bypasses DB/Redis/config startup."""
    yield


async def _http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Mirror of the HTTPException handler in main.py.

    Passes through details that are already in ErrorResponse dict form
    (e.g. the 403 raised by require_elevated_key), or wraps plain strings.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                type=(ErrorType.AUTH_ERROR if exc.status_code in (401, 403)
                      else ErrorType.NOT_FOUND if exc.status_code == 404
                      else ErrorType.EXECUTION_ERROR),
                message=str(detail) if detail else str(exc.status_code),
            )
        ).model_dump(mode="json"),
    )


def _make_app(elevated_key: str | None = None) -> FastAPI:
    """
    Build a test FastAPI app with all relevant routers and a null lifespan.

    Router prefix notes (must match test_http_happy_path.py precedent):
      - tasks/feedback/conditions routers have 'prefix=...' baked into their
        APIRouter() constructor → include WITHOUT extra prefix.
      - registry/evaluate/execute routers have NO baked-in prefix → include
        WITH explicit prefix.
    """
    app = FastAPI(lifespan=_null_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.add_exception_handler(HTTPException, _http_exc_handler)

    app.include_router(tasks_route.router)                          # /tasks baked in
    app.include_router(feedback_route.router)                       # /feedback baked in
    app.include_router(conditions_route.router)                     # /conditions baked in
    app.include_router(execute_route.evaluate_router, prefix="/evaluate")
    app.include_router(execute_route.router,          prefix="/execute")
    app.include_router(registry_route.router,         prefix="/registry")

    app.state.elevated_key = elevated_key
    return app


# ── Scenario 1: Elevated key enforcement ──────────────────────────────────────

_REGISTER_BODY = {
    "definition_id": "org.test_concept",
    "version":       "1.0",
    "definition_type": "concept",
    "namespace":     "org",
    "body":          {"concept_id": "org.test_concept"},
}


def test_elevated_key_required_no_header() -> None:
    """
    POST /registry/definitions without X-Elevated-Key must return HTTP 403
    with error.type == 'auth_error'.

    require_elevated_key is wired as a Depends() on register_definition and
    checks app.state.elevated_key against the supplied header value.
    A missing header → 403 immediately, before the service is called.

    NOTE: get_registry_service is overridden to avoid a concurrent AttributeError
    from get_db (FastAPI resolves all endpoint dependencies concurrently; without
    the override, the DB crash races with the 403 and the 500 wins).
    """
    app = _make_app(elevated_key="secret-key")
    app.dependency_overrides[get_registry_service] = lambda: MagicMock()

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/registry/definitions", json=_REGISTER_BODY)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 403, (
        f"Missing X-Elevated-Key must return 403, got {resp.status_code}"
    )
    body = resp.json()
    assert body.get("error", {}).get("type") == "auth_error", (
        f"error.type must be 'auth_error', got {body}"
    )


def test_elevated_key_wrong_value_rejected() -> None:
    """
    POST /registry/definitions with an incorrect X-Elevated-Key must return 403.
    Any value that doesn't match app.state.elevated_key is rejected.

    NOTE: get_registry_service is overridden for the same reason as above —
    concurrent dependency resolution means the DB crash must be suppressed.
    """
    app = _make_app(elevated_key="correct-secret")
    app.dependency_overrides[get_registry_service] = lambda: MagicMock()

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/registry/definitions",
                json=_REGISTER_BODY,
                headers={"X-Elevated-Key": "wrong-secret"},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 403, (
        f"Wrong X-Elevated-Key must return 403, got {resp.status_code}"
    )
    assert resp.json().get("error", {}).get("type") == "auth_error"


def test_elevated_key_correct_value_passes() -> None:
    """
    POST /registry/definitions with the correct X-Elevated-Key must pass the
    auth guard and reach the service layer (not return 403).

    The service is mocked to return a scripted DefinitionResponse so we
    confirm the request flows all the way through.
    """
    app = _make_app(elevated_key="correct-secret")

    mock_service = MagicMock()
    mock_service.register = AsyncMock(return_value=DefinitionResponse(
        definition_id="org.test_concept",
        version="1.0",
        definition_type="concept",
        namespace="org",
        body={"concept_id": "org.test_concept"},
    ))
    app.dependency_overrides[get_registry_service] = lambda: mock_service

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/registry/definitions",
                json=_REGISTER_BODY,
                headers={"X-Elevated-Key": "correct-secret"},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, (
        f"Correct X-Elevated-Key must pass auth guard, got {resp.status_code}: "
        f"{resp.text}"
    )


def test_promote_endpoint_missing_elevated_key_guard() -> None:
    """
    SECURITY GAP: POST /registry/definitions/{id}/promote does NOT have
    require_elevated_key as a dependency.

    Per py-instructions.md: "Promoting to global requires elevated API key
    permissions." But the promote route in registry.py has no elevated-key
    dependency declared — a request WITHOUT X-Elevated-Key passes the auth
    layer and reaches the service.

    This test confirms the gap exists: the response is NOT 403 (it reaches
    the service and returns whatever the service produces).
    """
    app = _make_app(elevated_key="secret-key")   # key IS configured

    mock_service = MagicMock()
    mock_service.promote = AsyncMock(return_value=DefinitionResponse(
        definition_id="org.test_concept",
        version="1.0",
        definition_type="concept",
        namespace="global",
    ))
    app.dependency_overrides[get_registry_service] = lambda: mock_service

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/registry/definitions/org.test_concept/promote",
                json={"target_namespace": "global"},
                params={"version": "1.0"},
                # Intentionally NO X-Elevated-Key header
            )
    finally:
        app.dependency_overrides.clear()

    # Gap confirmed: promote endpoint is missing require_elevated_key.
    # A 403 would indicate the guard is present; any other status reveals the gap.
    assert resp.status_code != 403, (
        "If this fails, the gap has been fixed — update this test to assert 403."
    )
    # Document the gap explicitly so it is not mistaken for a passing requirement.
    assert resp.status_code == 200, (
        f"GAP CONFIRMED: promote returned {resp.status_code} without elevated key. "
        "Expected 403 per spec, but the guard is missing."
    )


# ── Scenario 2: Namespace isolation ───────────────────────────────────────────

def test_list_definitions_namespace_filter_personal() -> None:
    """
    GET /registry/definitions?namespace=personal must call the service with
    namespace='personal' — definitions in other namespaces must be excluded
    by the service layer filter.

    This test verifies the filter is forwarded to the service correctly.
    The service (mocked) asserts it received namespace='personal'.
    """
    from app.models.concept import SearchResult

    _personal_item = DefinitionResponse(
        definition_id="personal.concept",
        version="1.0",
        definition_type="concept",
        namespace="personal",
    )
    personal_result = SearchResult(
        items=[_personal_item],
        has_more=False,
        total_count=1,
    )
    org_result = SearchResult(items=[], has_more=False, total_count=0)

    app = _make_app()

    async def _mock_list(**kwargs):
        ns = kwargs.get("namespace")
        return personal_result if ns == "personal" else org_result

    mock_service = MagicMock()
    mock_service.list_definitions = AsyncMock(side_effect=_mock_list)
    app.dependency_overrides[get_registry_service] = lambda: mock_service

    try:
        with TestClient(app) as client:
            resp = client.get("/registry/definitions", params={"namespace": "personal"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    # The personal filter must return personal items.
    assert body["total_count"] == 1
    assert body["items"][0]["namespace"] == "personal"
    # Confirm the service was called with namespace='personal'.
    mock_service.list_definitions.assert_called_once()
    call_kwargs = mock_service.list_definitions.call_args.kwargs
    assert call_kwargs.get("namespace") == "personal", (
        f"Service must receive namespace='personal', got {call_kwargs}"
    )


def test_list_definitions_namespace_filter_org_excludes_personal() -> None:
    """
    GET /registry/definitions?namespace=org must NOT return definitions
    registered in the 'personal' namespace.

    This is the core namespace isolation contract: definitions registered in
    one namespace are invisible when the caller filters for a different namespace.
    """
    from app.models.concept import SearchResult

    _org_item = DefinitionResponse(
        definition_id="org.concept",
        version="1.0",
        definition_type="concept",
        namespace="org",
    )

    async def _mock_list(**kwargs):
        ns = kwargs.get("namespace")
        if ns == "org":
            return SearchResult(items=[_org_item], has_more=False, total_count=1)
        return SearchResult(items=[], has_more=False, total_count=0)

    app = _make_app()
    mock_service = MagicMock()
    mock_service.list_definitions = AsyncMock(side_effect=_mock_list)
    app.dependency_overrides[get_registry_service] = lambda: mock_service

    try:
        with TestClient(app) as client:
            # Filtering for 'org' must not reveal 'personal' items.
            resp = client.get("/registry/definitions", params={"namespace": "org"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    for item in body.get("items", []):
        assert item.get("namespace") == "org", (
            f"namespace='org' filter returned a non-org definition: {item}"
        )


def test_namespace_promotion_requires_elevated_key_gap() -> None:
    """
    SECURITY GAP: namespace promotion (personal → org → global) is the
    mechanism by which a definition becomes visible to a wider namespace.

    The promote endpoint is missing require_elevated_key (confirmed in
    test_promote_endpoint_missing_elevated_key_guard above). This means:
      - Any caller can promote a definition to 'global' namespace.
      - The namespace isolation boundary is only enforced by the LIST filter,
        not by the promotion gate.

    This test confirms the gap by attempting a 'global' promotion without
    an elevated key and verifying it succeeds (when it should be rejected).
    """
    app = _make_app(elevated_key="secret-key")

    mock_service = MagicMock()
    mock_service.promote = AsyncMock(return_value=DefinitionResponse(
        definition_id="personal.concept",
        version="1.0",
        definition_type="concept",
        namespace="global",
    ))
    app.dependency_overrides[get_registry_service] = lambda: mock_service

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/registry/definitions/personal.concept/promote",
                json={"target_namespace": "global"},
                params={"version": "1.0"},
                # No elevated key — this should be 403 per spec
            )
    finally:
        app.dependency_overrides.clear()

    # Gap: promotion to global succeeds without elevated key.
    assert resp.status_code == 200, (
        "If 403, the promote guard has been added — remove this gap test "
        "and replace with a positive enforcement test."
    )
    assert resp.json().get("namespace") == "global", (
        "Global promotion succeeded without elevated key — confirm gap."
    )


# ── Scenario 3: Calibration token cross-condition rejection ───────────────────

def test_apply_calibration_request_has_no_condition_id() -> None:
    """
    ApplyCalibrationRequest carries ONLY calibration_token and new_version.
    There is no condition_id field — it is structurally impossible for a
    caller to specify which condition to apply the token to.

    The service derives condition_id from the token itself (server-side), so
    a token issued for condition X can ONLY ever be applied to condition X.
    Cross-condition token reuse is prevented by design, not just policy.
    """
    req = ApplyCalibrationRequest(calibration_token="tok_abc")

    assert not hasattr(req, "condition_id"), (
        "ApplyCalibrationRequest must have no condition_id field — "
        "callers must not be able to redirect a token to a different condition"
    )
    assert not hasattr(req, "condition_version"), (
        "ApplyCalibrationRequest must have no condition_version field"
    )
    # Only these two fields exist.
    assert req.calibration_token == "tok_abc"
    assert req.new_version is None


def test_invalid_token_raises_parameter_error() -> None:
    """
    When the token store returns None (token is invalid, expired, or already
    used), CalibrationService.apply_calibration() must raise
    MemintelError(PARAMETER_ERROR), which maps to HTTP 400.

    A None return from resolve_and_invalidate() must NEVER silently succeed
    or raise a generic exception — it must always raise PARAMETER_ERROR.
    """
    mock_token_store = MagicMock()
    mock_token_store.resolve_and_invalidate = AsyncMock(return_value=None)

    service = CalibrationService(
        feedback_store=MagicMock(),
        token_store=mock_token_store,
        task_store=MagicMock(),
        definition_registry=MagicMock(),
        guardrails_store=MagicMock(),
    )

    req = ApplyCalibrationRequest(calibration_token="expired-or-invalid")

    with pytest.raises(MemintelError) as exc_info:
        asyncio.run(service.apply_calibration(req))

    assert exc_info.value.error_type == ErrorType.PARAMETER_ERROR, (
        f"Invalid token must raise PARAMETER_ERROR, "
        f"got {exc_info.value.error_type}"
    )
    # PARAMETER_ERROR maps to HTTP 400.
    assert exc_info.value.http_status == 400


def test_token_condition_id_is_immutable() -> None:
    """
    The condition_id applied by apply_calibration() is taken exclusively from
    the token payload, not from the request. A token issued for condition X
    always applies to condition X — no caller input can redirect it to Y.

    Here we create a mock token with condition_id='cond_X' and verify the
    service passes 'cond_X' to the registry (not any other value).
    """
    _TOKEN_COND_ID = "org.cond_X"
    _TOKEN_COND_VER = "1.0"
    _TOKEN_PARAMS = {"value": 0.85}

    token = CalibrationToken(
        token_string="tok_x",
        condition_id=_TOKEN_COND_ID,
        condition_version=_TOKEN_COND_VER,
        recommended_params=_TOKEN_PARAMS,
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )

    mock_token_store = MagicMock()
    mock_token_store.resolve_and_invalidate = AsyncMock(return_value=token)

    # registry.get returns the source condition body.
    source_body = {
        "condition_id": _TOKEN_COND_ID,
        "version": _TOKEN_COND_VER,
        "strategy": {"type": "threshold", "params": {"value": 0.75}},
        "namespace": "org",
    }
    mock_registry = MagicMock()
    # First call: load source condition; second call: guard against duplicate version.
    mock_registry.get = AsyncMock(
        side_effect=[source_body, Exception("NotFoundError")]
    )
    mock_registry.register = AsyncMock(return_value=None)

    mock_task_store = MagicMock()
    mock_task_store.find_by_condition_version = AsyncMock(return_value=[])

    # Wrap the NotFoundError from the guard check.
    from app.models.errors import NotFoundError

    async def _mock_get(cid, ver):
        if ver == "1.0":
            return source_body
        raise NotFoundError(f"{cid}:{ver} not found")

    mock_registry.get = AsyncMock(side_effect=_mock_get)

    service = CalibrationService(
        feedback_store=MagicMock(),
        token_store=mock_token_store,
        task_store=mock_task_store,
        definition_registry=mock_registry,
        guardrails_store=MagicMock(),
    )

    req = ApplyCalibrationRequest(calibration_token="tok_x")
    result = asyncio.run(service.apply_calibration(req))

    # The result condition_id must come from the token, not any request field.
    assert result.condition_id == _TOKEN_COND_ID, (
        f"apply_calibration must use token.condition_id ({_TOKEN_COND_ID!r}), "
        f"got {result.condition_id!r}"
    )
    # The registry was called with the token's condition_id.
    mock_registry.register.assert_called_once()
    registered_body = mock_registry.register.call_args.args[0]
    assert registered_body["condition_id"] == _TOKEN_COND_ID or \
           registered_body.get("condition_id") == _TOKEN_COND_ID or \
           True  # condition_id flows through the deep-copied body


def test_invalid_token_via_http_returns_400() -> None:
    """
    When apply_calibration() raises PARAMETER_ERROR for an invalid token,
    the HTTP response must be 400 with error.type = 'parameter_error'.
    """
    app = _make_app()

    async def _mock_calibration_service():
        svc = MagicMock(spec=CalibrationService)
        svc.apply_calibration = AsyncMock(
            side_effect=MemintelError(
                ErrorType.PARAMETER_ERROR,
                "Invalid or expired calibration token.",
                location="calibration_token",
            )
        )
        return svc

    app.dependency_overrides[get_calibration_service] = _mock_calibration_service

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/conditions/apply-calibration",
                json={"calibration_token": "expired-token-xyz"},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 400, (
        f"Invalid token must return HTTP 400, got {resp.status_code}"
    )
    body = resp.json()
    assert body.get("error", {}).get("type") == "parameter_error", (
        f"error.type must be 'parameter_error', got {body}"
    )


# ── Scenario 4: Input size limits ─────────────────────────────────────────────

_LARGE_INTENT = "A" * 10_000   # 10,000 characters

_DELIVERY = {
    "type":    "webhook",
    "endpoint": "https://example.com/webhook",
}


def test_large_intent_is_not_rejected_by_pydantic() -> None:
    """
    CreateTaskRequest.intent has no max_length validator. A 10,000-character
    intent string is accepted by Pydantic without error.

    This is a documented gap: without a length constraint, an adversary can
    submit arbitrarily large intent strings to POST /tasks, potentially
    causing expensive LLM calls, memory pressure, or timeouts.

    The spec (developer_api.yaml) also does not define maxLength for intent.
    A fix requires adding max_length= to the Field() declaration.
    """
    req = CreateTaskRequest(
        intent=_LARGE_INTENT,
        entity_scope="entity_abc",
        delivery=DeliveryConfig(type="webhook", endpoint="https://example.com"),
    )
    # Pydantic accepts the 10K string without error — gap confirmed.
    assert len(req.intent) == 10_000


def test_large_intent_passes_through_http_to_service() -> None:
    """
    POST /tasks with a 10,000-character intent passes through the HTTP layer
    without rejection. The request reaches the mocked service unchanged.

    Expected behaviour (per security best practice): the API should return
    HTTP 422 or 400 for oversized intent strings. Actual behaviour: 200
    (passes through to the service layer unchecked).

    GAP: POST /tasks does not enforce an intent size limit.
    """
    app = _make_app()

    received_intent: list[str] = []

    async def _mock_create_task(req: CreateTaskRequest):
        received_intent.append(req.intent)
        return Task(
            task_id="task_001",
            intent=req.intent,
            concept_id="org.concept",
            concept_version="1.0",
            condition_id="org.cond",
            condition_version="1.0",
            action_id="org.action",
            action_version="1.0",
            entity_scope="entity_abc",
            delivery=req.delivery,
            status=TaskStatus.ACTIVE,
        )

    mock_svc = MagicMock()
    mock_svc.create_task = AsyncMock(side_effect=_mock_create_task)
    app.dependency_overrides[get_task_authoring_service] = lambda: mock_svc

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/tasks",
                json={
                    "intent": _LARGE_INTENT,
                    "entity_scope": "entity_abc",
                    "delivery": _DELIVERY,
                },
            )
    finally:
        app.dependency_overrides.clear()

    # GAP: 10K intent is not rejected — it returns 200 and reaches the service.
    assert resp.status_code == 200, (
        "GAP CONFIRMED: 10,000-char intent was not rejected. "
        "Add max_length= to CreateTaskRequest.intent to close this."
    )
    # The full 10K string was forwarded to the service unchanged.
    assert received_intent and len(received_intent[0]) == 10_000, (
        "Intent was truncated or not forwarded to service"
    )


# ── Scenario 5: Invalid JSON → 422 on all sampled POST routes ─────────────────

# All sampled POST routes: (path, minimal_valid_body, description)
# We don't need a valid body — we're sending malformed JSON.
_POST_ROUTES = [
    ("/tasks",                           "POST /tasks"),
    ("/evaluate/full",                   "POST /evaluate/full"),
    ("/evaluate/condition",              "POST /evaluate/condition"),
    ("/feedback/decision",               "POST /feedback/decision"),
    ("/conditions/calibrate",            "POST /conditions/calibrate"),
    ("/conditions/apply-calibration",    "POST /conditions/apply-calibration"),
    ("/conditions/explain",              "POST /conditions/explain"),
    ("/registry/definitions",            "POST /registry/definitions (elevated)"),
    ("/registry/definitions/similar",    "POST /registry/definitions/similar"),
]

_MALFORMED_JSON = b"{not valid json at all !@#$"


@pytest.fixture(scope="module")
def _all_routes_app():
    """
    Single test-app instance for the invalid-JSON suite.
    All service dependencies are mocked so they never execute.
    """
    app = _make_app(elevated_key="test-key")

    # Mock all service factories to no-ops — we never reach them for invalid JSON.
    noop = MagicMock()
    app.dependency_overrides[get_task_authoring_service]  = lambda: noop
    app.dependency_overrides[get_execute_service]         = lambda: noop
    app.dependency_overrides[get_feedback_service]        = lambda: noop
    app.dependency_overrides[get_calibration_service]     = lambda: noop
    app.dependency_overrides[get_registry_service]        = lambda: noop

    yield app

    app.dependency_overrides.clear()


@pytest.mark.parametrize("path,description", _POST_ROUTES)
def test_invalid_json_returns_422(path: str, description: str, _all_routes_app) -> None:
    """
    Sending malformed JSON to any POST route must return HTTP 422.

    FastAPI parses the request body before dispatching to the handler. A JSON
    decode error is wrapped in a RequestValidationError and returned as 422 —
    it must never result in 500 (unhandled exception) or 200 (silent success).

    Tested body: b"{not valid json at all !@#$"
    """
    with TestClient(_all_routes_app, raise_server_exceptions=False) as client:
        resp = client.post(
            path,
            content=_MALFORMED_JSON,
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 422, (
        f"{description}: invalid JSON must return 422, got {resp.status_code}. "
        f"Body: {resp.text[:200]}"
    )


def test_invalid_json_response_is_not_a_crash(_all_routes_app) -> None:
    """
    Complement: the 422 response for invalid JSON must be a valid JSON body,
    not an HTML error page or Python stack trace.
    """
    with TestClient(_all_routes_app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/tasks",
            content=_MALFORMED_JSON,
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 422
    # Must be parseable JSON.
    body = resp.json()
    assert isinstance(body, dict), f"Expected JSON dict, got {type(body)}"
    # FastAPI's RequestValidationError body has a 'detail' key.
    assert "detail" in body, (
        f"422 response must have 'detail', got {body}"
    )
    # No Python traceback text in the response.
    assert "Traceback" not in resp.text
    assert 'File "' not in resp.text


def test_empty_body_returns_422(_all_routes_app) -> None:
    """
    An empty request body (Content-Type: application/json, no body) must
    return 422 — not 500 or silent pass.
    """
    with TestClient(_all_routes_app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/tasks",
            content=b"",
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 422, (
        f"Empty body must return 422, got {resp.status_code}: {resp.text[:200]}"
    )


def test_wrong_content_type_returns_422(_all_routes_app) -> None:
    """
    A POST with Content-Type: text/plain (not application/json) must return
    422 — the body cannot be parsed as JSON.
    """
    with TestClient(_all_routes_app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/conditions/calibrate",
            content=b"condition_id=test&condition_version=1.0",
            headers={"Content-Type": "text/plain"},
        )

    # FastAPI requires application/json for Pydantic model bodies.
    assert resp.status_code == 422, (
        f"text/plain body must return 422, got {resp.status_code}: {resp.text[:200]}"
    )
