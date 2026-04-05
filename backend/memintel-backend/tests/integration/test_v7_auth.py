"""
tests/integration/test_v7_auth.py
──────────────────────────────────────────────────────────────────────────────
T-4 Part 2 — Auth boundary integration tests.

Verifies the ACTUAL enforcement mechanism (app/api/deps.py):

  require_api_key (X-Api-Key header)
    Applied to: POST /concepts/compile, POST /concepts/register
    Permissive when app.state.api_key is None (dev mode).
    Returns HTTP 401 when key is configured but missing/wrong.

  require_elevated_key (X-Elevated-Key header)
    Always enforced — returns 403 even when no key is configured.
    NOT applied to any V7 concept/task route (tested via a direct
    minimal endpoint).

  POST /tasks (create_task) — NO auth dependency at all.
    This is a deliberate architectural choice (Canvas handles its own
    session auth). Standard and elevated keys are not checked.

  Identity fields (user_id, org_id, module_id) — ignored silently
    per Dev Guide section 4.1. Extra Pydantic fields are dropped.

Tests
─────
  test_compile_requires_standard_key
  test_register_requires_standard_key
  test_create_task_has_no_auth
  test_elevated_key_dependency_enforcement
  test_identity_fields_ignored
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from fastapi import Depends, FastAPI, Header, Request

from app.api.deps import require_api_key, require_elevated_key
from app.api.routes import concepts as concepts_route
from app.api.routes import tasks as tasks_route
from app.api.routes.concepts import (
    get_concept_compiler_service,
    get_concept_registration_service,
)
from app.api.routes.tasks import get_task_authoring_service
from app.models.errors import MemintelError, memintel_error_handler
from app.registry.definitions import DefinitionRegistry
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.services.task_authoring import TaskAuthoringService
from app.stores import DefinitionStore, TaskStore
from app.stores.compile_token import CompileTokenStore

from tests.integration.conftest_v7 import LLMMockClient


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_auth_app(
    db_pool,
    llm_client: Any,
    *,
    api_key: str | None = None,
    elevated_key: str | None = None,
) -> FastAPI:
    """
    Minimal FastAPI app with configurable API keys.

    api_key      — when set, require_api_key enforces X-Api-Key.
    elevated_key — when set, require_elevated_key enforces X-Elevated-Key.
    """
    app = FastAPI()
    app.state.db = db_pool
    if api_key is not None:
        app.state.api_key = api_key
    if elevated_key is not None:
        app.state.elevated_key = elevated_key
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)
    app.include_router(tasks_route.router)

    async def _compiler_svc() -> ConceptCompilerService:
        return ConceptCompilerService(
            llm_client=llm_client,
            token_store=CompileTokenStore(db_pool),
        )

    async def _registration_svc() -> ConceptRegistrationService:
        return ConceptRegistrationService()

    async def _task_svc() -> TaskAuthoringService:
        return TaskAuthoringService(
            task_store=TaskStore(db_pool),
            definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
            llm_client=llm_client,
        )

    app.dependency_overrides[get_concept_compiler_service] = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service] = _task_svc
    return app


_COMPILE_BODY = {
    "identifier":      "auth.test_concept",
    "description":     "Auth boundary test concept",
    "output_type":     "float",
    "signal_names":    ["test_signal"],
    "stream":          False,
    "return_reasoning": False,
}

_TASK_BODY = {
    "intent":       "alert when loan repayment ratio is below 0.80",
    "entity_scope": "loan",
    "delivery": {
        "type":     "webhook",
        "endpoint": "https://auth-test.example.com/hook",
    },
    "stream":   False,
    "dry_run":  True,   # avoid persisting tasks
}


# ═══════════════════════════════════════════════════════════════════════════════
# test_compile_requires_standard_key
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompileRequiresStandardKey:
    """
    POST /concepts/compile uses require_api_key (X-Api-Key header).
    Returns 401 when key is configured but missing or wrong.
    Returns 201 with a valid key.
    """

    def test_compile_requires_standard_key(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_auth_app(db_pool, llm, api_key="test-standard-key-001")
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # No key → 401
                resp = await client.post("/concepts/compile", json=_COMPILE_BODY)
                assert resp.status_code == 401, (
                    f"Expected 401 with no key, got {resp.status_code}: {resp.text}"
                )
                # HTTPException wraps detail in {"detail": {...}}
                body = resp.json()
                err = body.get("detail", body)
                assert "error" in err, f"No 'error' in response: {body}"
                assert err["error"]["type"] == "auth_error"

                # Wrong key → 401
                resp = await client.post(
                    "/concepts/compile",
                    json=_COMPILE_BODY,
                    headers={"X-Api-Key": "wrong-key"},
                )
                assert resp.status_code == 401, (
                    f"Expected 401 with wrong key, got {resp.status_code}"
                )

                # Valid key → 201
                resp = await client.post(
                    "/concepts/compile",
                    json=_COMPILE_BODY,
                    headers={"X-Api-Key": "test-standard-key-001"},
                )
                assert resp.status_code == 201, (
                    f"Expected 201 with valid key, got {resp.status_code}: {resp.text}"
                )
                body = resp.json()
                assert "compile_token" in body

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_register_requires_standard_key
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegisterRequiresStandardKey:
    """
    POST /concepts/register uses require_api_key (X-Api-Key header).
    NOTE: Both compile and register use require_api_key — NOT elevated key.
    The elevated_key mechanism guards different routes (guardrails, registry writes).

    Returns 401 when key is configured but missing or wrong.
    Returns 201 with a valid key.
    """

    def test_register_requires_standard_key(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_auth_app(db_pool, llm, api_key="test-standard-key-002")
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # First compile with valid key to get a token
                compile_resp = await client.post(
                    "/concepts/compile",
                    json={
                        **_COMPILE_BODY,
                        "identifier": "auth.register_test",
                    },
                    headers={"X-Api-Key": "test-standard-key-002"},
                )
                assert compile_resp.status_code == 201, compile_resp.text
                token = compile_resp.json()["compile_token"]

                register_body = {
                    "compile_token": token,
                    "identifier":    "auth.register_test",
                }

                # No key → 401
                resp = await client.post("/concepts/register", json=register_body)
                assert resp.status_code == 401, (
                    f"Expected 401 with no key, got {resp.status_code}: {resp.text}"
                )
                body = resp.json()
                err = body.get("detail", body)
                assert "error" in err, f"No 'error' in response: {body}"
                assert err["error"]["type"] == "auth_error"

                # Wrong key → 401
                resp = await client.post(
                    "/concepts/register",
                    json=register_body,
                    headers={"X-Api-Key": "wrong-key"},
                )
                assert resp.status_code == 401, (
                    f"Expected 401 with wrong key, got {resp.status_code}"
                )

                # Valid key → 201
                resp = await client.post(
                    "/concepts/register",
                    json=register_body,
                    headers={"X-Api-Key": "test-standard-key-002"},
                )
                assert resp.status_code == 201, (
                    f"Expected 201 with valid key, got {resp.status_code}: {resp.text}"
                )
                assert "concept_id" in resp.json()

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_create_task_has_no_auth
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateTaskHasNoAuth:
    """
    POST /tasks (create_task) has NO auth dependency.

    This is intentional: Canvas owns session auth; Memintel trusts the
    caller at the task-creation boundary. Confirm that even when
    app.state.api_key is set, POST /tasks is accessible without any key.
    """

    def test_create_task_has_no_auth(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_auth_app(db_pool, llm, api_key="must-not-block-tasks")
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # POST /tasks with NO key, even though api_key is configured
                resp = await client.post("/tasks", json=_TASK_BODY)
                assert resp.status_code == 200, (
                    f"POST /tasks should succeed without any key — got {resp.status_code}: {resp.text}"
                )
                body = resp.json()
                # dry_run=True → DryRunResult shape (no task_id)
                assert "concept" in body or "action_id" in body, (
                    f"Unexpected response shape: {body}"
                )

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_elevated_key_dependency_enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestElevatedKeyDependencyEnforcement:
    """
    require_elevated_key (X-Elevated-Key header) is ALWAYS enforced:
      - 403 when header is absent
      - 403 when header value is wrong
      - 403 when app.state.elevated_key is None (not configured)
      - 200 when header matches app.state.elevated_key

    Tested via a minimal FastAPI endpoint that uses require_elevated_key
    directly — independent of the DB pool.
    """

    def test_elevated_key_dependency_enforcement(self, run):
        # Minimal app with one elevated-key-guarded endpoint
        elev_app = FastAPI()
        elev_app.state.elevated_key = "elev-secret-test-003"

        @elev_app.get("/elevated-test")
        async def _guarded(
            request: Request,
            x_elevated_key: str | None = Header(default=None),
        ) -> dict:
            await require_elevated_key(request, x_elevated_key)
            return {"ok": True}

        transport = httpx.ASGITransport(app=elev_app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # No key → 403
                resp = await client.get("/elevated-test")
                assert resp.status_code == 403, (
                    f"Expected 403 with no elevated key, got {resp.status_code}"
                )
                body = resp.json()
                err = body.get("detail", body)
                assert "error" in err, f"No 'error' in response: {body}"
                assert err["error"]["type"] == "auth_error"

                # Wrong key → 403
                resp = await client.get(
                    "/elevated-test",
                    headers={"X-Elevated-Key": "wrong-elevated-key"},
                )
                assert resp.status_code == 403, (
                    f"Expected 403 with wrong elevated key, got {resp.status_code}"
                )

                # Correct key → 200
                resp = await client.get(
                    "/elevated-test",
                    headers={"X-Elevated-Key": "elev-secret-test-003"},
                )
                assert resp.status_code == 200, (
                    f"Expected 200 with correct elevated key, got {resp.status_code}"
                )
                assert resp.json() == {"ok": True}

        run(_go())

    def test_elevated_key_403_when_not_configured(self, run):
        """
        require_elevated_key raises 403 even when app.state.elevated_key is None.
        This differs from require_api_key which is permissive when unconfigured.
        """
        elev_app = FastAPI()
        # Note: NOT setting elev_app.state.elevated_key

        @elev_app.get("/elevated-unconfigured")
        async def _guarded(
            request: Request,
            x_elevated_key: str | None = Header(default=None),
        ) -> dict:
            await require_elevated_key(request, x_elevated_key)
            return {"ok": True}

        transport = httpx.ASGITransport(app=elev_app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Even with a key header, 403 because no key is configured server-side
                resp = await client.get(
                    "/elevated-unconfigured",
                    headers={"X-Elevated-Key": "any-key"},
                )
                assert resp.status_code == 403, (
                    f"Expected 403 when elevated_key not configured, got {resp.status_code}"
                )

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_identity_fields_ignored
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityFieldsIgnored:
    """
    Per Dev Guide section 4.1: Canvas may send identity context fields
    (user_id, org_id, module_id) that Memintel MUST silently ignore.

    These fields are not in CreateTaskRequest and Pydantic's default
    extra-field handling drops them. The request must succeed and the
    response must not echo them back.
    """

    def test_identity_fields_ignored(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_auth_app(db_pool, llm)  # no api_key → permissive
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                body_with_identity = {
                    **_TASK_BODY,
                    "user_id":   "usr_test_001",
                    "org_id":    "org_test_001",
                    "module_id": "mod_test_001",
                }
                resp = await client.post("/tasks", json=body_with_identity)
                assert resp.status_code == 200, (
                    f"Task with identity fields should succeed — got {resp.status_code}: {resp.text}"
                )
                response_body = resp.json()

                # Identity fields must NOT appear in the response
                assert "user_id" not in response_body, (
                    "Response must not echo user_id"
                )
                assert "org_id" not in response_body, (
                    "Response must not echo org_id"
                )
                assert "module_id" not in response_body, (
                    "Response must not echo module_id"
                )

        run(_go())
