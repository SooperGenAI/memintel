"""
tests/integration/test_v7_sse_disconnect.py
──────────────────────────────────────────────────────────────────────────────
T-4 Part 4 — SSE client-disconnect robustness tests.

Validates that the server remains healthy after a client disconnects mid-stream
(i.e., the client stops consuming SSE events before the stream completes).

Design
──────
With httpx.ASGITransport + client.stream(), a disconnect is simulated by
breaking out of the response iteration loop and exiting the context manager.
The server-side async generator is abandoned and eventually garbage-collected.

"Server health" is verified by making a subsequent normal (non-streaming)
request to the same app instance and asserting it succeeds.

We deliberately do NOT assert on the internal generator state after disconnect
(unobservable from the outside). All tests focus on the observable behaviour:
  - The server returns a valid response to the NEXT request.
  - No exception escapes to the test.
  - For dry_run=True tasks: the DB has no task rows after a mid-stream abort.

Tests
─────
  test_disconnect_after_first_step
    /tasks stream=True — read one SSE line, close, verify server healthy.

  test_disconnect_after_compile_first_step
    /concepts/compile stream=True — read one SSE line, close, verify healthy.

  test_disconnect_midstream_no_orphaned_task
    /tasks stream=True dry_run=True — break after 2 events, verify no task
    rows exist in the DB (dry_run never persists regardless of disconnect).

  test_server_healthy_after_multiple_disconnects
    5 sequential stream-open-and-close cycles, then one normal non-streaming
    POST /tasks — must succeed with HTTP 200.
"""
from __future__ import annotations

from typing import Any

import asyncpg
import httpx
import pytest

from fastapi import FastAPI

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

def _make_sse_app(db_pool: asyncpg.Pool, llm_client: Any) -> FastAPI:
    """
    Minimal FastAPI app for SSE disconnect tests.

    No api_key configured → permissive mode (no auth headers needed).
    """
    app = FastAPI()
    app.state.db = db_pool
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


# ── Shared request bodies ──────────────────────────────────────────────────────

_STREAM_TASK_BODY = {
    "intent":       "alert when loan repayment ratio is below 0.80",
    "entity_scope": "loan",
    "delivery": {
        "type":     "webhook",
        "endpoint": "https://sse-test.example.com/hook",
    },
    "stream":   True,
    "dry_run":  True,   # avoid persisting tasks in all but one test
}

_STREAM_COMPILE_BODY = {
    "identifier":       "sse.disconnect_test",
    "description":      "SSE disconnect test concept",
    "output_type":      "float",
    "signal_names":     ["payments_on_time", "payments_due"],
    "stream":           True,
    "return_reasoning": False,
}

_NORMAL_TASK_BODY = {
    "intent":       "alert when loan repayment ratio is below 0.80",
    "entity_scope": "loan",
    "delivery": {
        "type":     "webhook",
        "endpoint": "https://sse-test.example.com/hook",
    },
    "stream":  False,
    "dry_run": True,
}


# ═══════════════════════════════════════════════════════════════════════════════
# test_disconnect_after_first_step
# ═══════════════════════════════════════════════════════════════════════════════

class TestDisconnectAfterFirstStep:
    """
    Open a /tasks SSE stream, consume only the first event line, then
    close the response.  The server must remain healthy — verified by
    issuing a follow-up non-streaming request that returns HTTP 200.
    """

    def test_disconnect_after_first_step(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_sse_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # ── Simulate disconnect after first SSE line ───────────────────
                async with client.stream(
                    "POST", "/tasks", json=_STREAM_TASK_BODY
                ) as response:
                    assert response.status_code == 200, (
                        f"Expected 200, got {response.status_code}"
                    )
                    # Read at most one non-empty line then break (simulate disconnect)
                    async for line in response.aiter_lines():
                        if line.strip():
                            break  # got first content line, disconnect now

                # ── Server-health check — new non-streaming request ────────────
                health = await client.post("/tasks", json=_NORMAL_TASK_BODY)
                assert health.status_code == 200, (
                    f"Server unhealthy after SSE disconnect — got "
                    f"{health.status_code}: {health.text}"
                )
                body = health.json()
                # DryRunResult must have concept key
                assert "concept" in body or "action_id" in body, (
                    f"Unexpected response after disconnect: {body}"
                )

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_disconnect_after_compile_first_step
# ═══════════════════════════════════════════════════════════════════════════════

class TestDisconnectAfterCompileFirstStep:
    """
    Open a /concepts/compile SSE stream, consume only the first event
    line, then close.  The server must remain healthy.
    """

    def test_disconnect_after_compile_first_step(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_sse_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # ── Compile stream: break after first non-empty line ───────────
                async with client.stream(
                    "POST", "/concepts/compile", json=_STREAM_COMPILE_BODY
                ) as response:
                    assert response.status_code == 200, (
                        f"Expected 200 for compile stream, got {response.status_code}"
                    )
                    async for line in response.aiter_lines():
                        if line.strip():
                            break

                # ── Health check: non-streaming compile should still work ──────
                health = await client.post(
                    "/concepts/compile",
                    json={**_STREAM_COMPILE_BODY, "stream": False},
                )
                assert health.status_code == 201, (
                    f"Server unhealthy after compile SSE disconnect — got "
                    f"{health.status_code}: {health.text}"
                )
                assert "compile_token" in health.json(), (
                    f"compile_token missing from healthy response: {health.json()}"
                )

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_disconnect_midstream_no_orphaned_task
# ═══════════════════════════════════════════════════════════════════════════════

class TestDisconnectMidstreamNoOrphanedTask:
    """
    Disconnect after 2 SSE events from a dry_run=True /tasks stream.

    dry_run=True guarantees no task row is ever written to the DB, so
    zero tasks must remain in the DB after the disconnect regardless of
    when exactly the client closed the connection.

    Verified by checking that GET /tasks returns an empty list.
    """

    def test_disconnect_midstream_no_orphaned_task(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_sse_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # ── Stream 2 content lines then abort ─────────────────────────
                content_lines_read = 0
                async with client.stream(
                    "POST", "/tasks", json=_STREAM_TASK_BODY
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.strip():
                            content_lines_read += 1
                        if content_lines_read >= 2:
                            break  # mid-stream disconnect

                assert content_lines_read >= 1, (
                    "Expected to read at least 1 SSE content line before disconnect"
                )

                # ── Verify no task row persisted (dry_run guarantee) ───────────
                async with db_pool.acquire() as conn:
                    count = await conn.fetchval("SELECT COUNT(*) FROM tasks")
                assert count == 0, (
                    f"Expected 0 task rows after dry_run disconnect, found {count}"
                )

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_server_healthy_after_multiple_disconnects
# ═══════════════════════════════════════════════════════════════════════════════

class TestServerHealthyAfterMultipleDisconnects:
    """
    Simulate 5 sequential SSE disconnects (each reading only the first event)
    then issue one normal non-streaming POST /tasks.

    The final request must succeed with HTTP 200 and return a valid DryRunResult.
    This verifies that repeated early client closes do not corrupt any shared
    server-side state (connection pool, LLM client, task store).
    """

    def test_server_healthy_after_multiple_disconnects(self, db_pool, run):
        llm = LLMMockClient()
        app = _make_sse_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # ── 5 sequential early-disconnect SSE requests ─────────────────
                for i in range(5):
                    async with client.stream(
                        "POST", "/tasks", json=_STREAM_TASK_BODY
                    ) as response:
                        assert response.status_code == 200, (
                            f"Disconnect cycle {i+1}: expected 200, "
                            f"got {response.status_code}"
                        )
                        # Read at most one content line then abandon the stream
                        async for line in response.aiter_lines():
                            if line.strip():
                                break

                # ── Final health check: full non-streaming request ─────────────
                final = await client.post("/tasks", json=_NORMAL_TASK_BODY)
                assert final.status_code == 200, (
                    f"Server unhealthy after 5 SSE disconnects — "
                    f"got {final.status_code}: {final.text}"
                )
                body = final.json()
                assert "concept" in body or "action_id" in body, (
                    f"Unexpected response shape after 5 disconnects: {body}"
                )

        run(_go())
