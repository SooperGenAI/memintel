"""
tests/e2e/conftest.py
──────────────────────────────────────────────────────────────────────────────
E2E test infrastructure — full-stack TestClient wired to a real test database.

Design decisions
────────────────
1. Non-destructive DB setup: We CREATE IF NOT EXISTS rather than DROP+CREATE
   so running e2e tests alongside contract tests doesn't destroy the DB.
   Alembic `upgrade head` is idempotent and safe to run repeatedly.

2. No lifespan truncation: _make_e2e_app() does NOT call _truncate_all()
   inside the lifespan (unlike the contract test app). Truncation is handled
   by the e2e_setup fixture BEFORE the TestClient starts, giving tests full
   control over data lifecycle.

3. Two event loops: TestClient's anyio runs in a background thread with its
   own event loop. Direct DB operations (seeding rows, querying results) use
   a separate asyncpg pool with a separate event loop in the main thread.
   The two pools are completely independent — no "Future attached to a
   different loop" errors.

4. Task seeding: POST /tasks calls the LLM via TaskAuthoringService. E2E
   tests bypass this by inserting task rows directly via the `run_db` helper
   (the separate pool). The row schema matches TaskStore exactly.

Database: postgresql://postgres:admin@localhost:5433/memintel_test
Tests are skipped gracefully if the database is unavailable.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Coroutine
from unittest.mock import AsyncMock, MagicMock

# Stub aioredis before any app import — prevents ImportError in test context.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import asyncpg
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from app.api.routes import (
    actions,
    agents,
    compile,
    conditions,
    context,
    decisions,
    execute,
    feedback,
    guardrails_api,
    jobs,
    registry,
    tasks,
)
from app.config import PrimitiveRegistry
from app.models.errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorType,
    MemintelError,
    memintel_error_handler,
)

# ── Constants ──────────────────────────────────────────────────────────────────

TEST_DATABASE_URL: str = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:admin@localhost:5433/memintel_test",
)

ELEVATED_KEY = "contract-test-elevated-key"
API_KEY = "contract-test-api-key"

_ALL_TABLES: tuple[str, ...] = (
    "decisions",
    "concept_results",
    "feedback_records",
    "calibration_tokens",
    "execution_graphs",
    "jobs",
    "application_context",
    "guardrails_versions",
    "tasks",
    "definitions",
)


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _admin_url() -> str:
    base, _ = TEST_DATABASE_URL.rsplit("/", 1)
    return f"{base}/postgres"


async def _ensure_test_database() -> None:
    """Create memintel_test if it doesn't exist. Non-destructive."""
    conn = await asyncpg.connect(_admin_url())
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'memintel_test'"
        )
        if not exists:
            await conn.execute("CREATE DATABASE memintel_test")
    finally:
        await conn.close()


def _run_alembic_migrations() -> None:
    env = {**os.environ, "DATABASE_URL": TEST_DATABASE_URL}
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed (exit {result.returncode}):\n{result.stderr}"
        )


async def _truncate_all(pool: asyncpg.Pool) -> None:
    table_list = ", ".join(_ALL_TABLES)
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table_list} RESTART IDENTITY CASCADE")


# ── Session-level DB setup ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def _e2e_database_setup():
    """
    Ensure the memintel_test database exists and is migrated.

    Non-destructive: uses CREATE IF NOT EXISTS + idempotent alembic upgrade.
    Skips all e2e tests gracefully if PostgreSQL is unreachable.
    """
    setup_loop = asyncio.new_event_loop()
    try:
        setup_loop.run_until_complete(_ensure_test_database())
    except Exception as exc:
        setup_loop.close()
        pytest.skip(
            f"Cannot reach PostgreSQL at {_admin_url()} — is it running? Error: {exc}"
        )
        return
    finally:
        if not setup_loop.is_closed():
            setup_loop.close()

    try:
        _run_alembic_migrations()
    except Exception as exc:
        pytest.skip(f"Alembic migrations failed: {exc}")
        return

    yield


# ── Exception handler (mirrors main.py) ───────────────────────────────────────

async def _http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                type=(
                    ErrorType.AUTH_ERROR if exc.status_code in (401, 403)
                    else ErrorType.NOT_FOUND if exc.status_code == 404
                    else ErrorType.EXECUTION_ERROR
                ),
                message=str(detail) if detail else str(exc.status_code),
            )
        ).model_dump(mode="json"),
    )


# ── App factory (no lifespan truncation) ──────────────────────────────────────

def _make_e2e_app() -> FastAPI:
    """
    Build a FastAPI app with all production routers.

    Key difference from contract tests: we do NOT call _truncate_all() in the
    lifespan. The e2e_setup fixture truncates tables before the TestClient
    starts — this is the authoritative cleanup point.
    """
    @asynccontextmanager
    async def _e2e_lifespan(app: FastAPI):
        try:
            pool = await asyncpg.create_pool(
                TEST_DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to test database: {exc}") from exc

        primitive_registry = PrimitiveRegistry()
        redis_stub = MagicMock()
        redis_stub.close = AsyncMock()

        app.state.db = pool
        app.state.elevated_key = ELEVATED_KEY
        app.state.api_key = API_KEY
        app.state.guardrails_store = None
        app.state.primitive_registry = primitive_registry
        app.state.redis = redis_stub
        app.state.connector_registry = None
        app.state.config = None

        yield

        await pool.close()

    app = FastAPI(title="Memintel E2E Test API", lifespan=_e2e_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.add_exception_handler(HTTPException, _http_exc_handler)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(asyncpg.PostgresError)
    async def _postgres_handler(request: Request, exc: asyncpg.PostgresError) -> JSONResponse:
        if isinstance(exc, asyncpg.CheckViolationError):
            return JSONResponse(status_code=422, content={"detail": "Invalid field value"})
        if isinstance(exc, asyncpg.DataError):
            return JSONResponse(status_code=422, content={"detail": "Invalid data format"})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(execute.evaluate_router, prefix="/evaluate", tags=["Execution"])
    app.include_router(execute.router,          prefix="/execute",  tags=["Execution"])
    app.include_router(compile.router,          prefix="/compile",  tags=["Compiler"])
    app.include_router(registry.router,         prefix="/registry", tags=["Registry"])
    app.include_router(agents.router,           prefix="/agents",   tags=["Agents"])
    app.include_router(tasks.router,                                tags=["Tasks"])
    app.include_router(conditions.router,                           tags=["Conditions"])
    app.include_router(decisions.router,                            tags=["Decisions"])
    app.include_router(feedback.router,                             tags=["Feedback"])
    app.include_router(actions.router,                              tags=["Actions"])
    app.include_router(jobs.router,             prefix="/jobs",     tags=["Jobs"])
    app.include_router(context.router,          prefix="/context",  tags=["Context"])
    app.include_router(guardrails_api.router,                       tags=["Guardrails"])

    return app


# ── Core e2e fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def e2e_setup(_e2e_database_setup):
    """
    Function-scoped fixture providing a clean DB + helper for direct SQL.

    Yields (pool, run_db) where:
      pool   — asyncpg.Pool connected to the test DB (separate from TestClient)
      run_db — callable: run_db(coro) → runs coro on the fixture's event loop

    The pool runs in the fixture's own event loop (main thread), completely
    separate from TestClient's anyio event loop (background thread).
    Use run_db() for seeding rows and querying results directly.
    """
    loop = asyncio.new_event_loop()

    async def _setup():
        pool = await asyncpg.create_pool(
            TEST_DATABASE_URL,
            min_size=1,
            max_size=3,
            command_timeout=30,
        )
        await _truncate_all(pool)
        return pool

    try:
        pool = loop.run_until_complete(_setup())
    except Exception as exc:
        loop.close()
        pytest.skip(f"Test database unavailable: {exc}")
        return

    def run_db(coro: Coroutine) -> Any:
        return loop.run_until_complete(coro)

    yield pool, run_db

    async def _teardown():
        await pool.close()

    loop.run_until_complete(_teardown())
    loop.close()


@pytest.fixture
def e2e_client(e2e_setup):
    """
    Function-scoped fixture — yields (client, pool, run_db).

    client  — Starlette TestClient with full HTTP stack
    pool    — asyncpg pool for direct DB access (seeding/querying)
    run_db  — helper: run_db(coro) executes async SQL in the fixture loop
    """
    pool, run_db = e2e_setup
    app = _make_e2e_app()
    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, pool, run_db
    except RuntimeError as exc:
        if "Cannot connect" in str(exc):
            pytest.skip(f"Test database unavailable: {exc}")
            return
        raise


# ── Auth header fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def api_headers() -> dict[str, str]:
    """API key header for developer-facing read endpoints."""
    return {"X-Api-Key": API_KEY}


@pytest.fixture
def elevated_headers() -> dict[str, str]:
    """Elevated key header for internal platform endpoints."""
    return {"X-Elevated-Key": ELEVATED_KEY}


@pytest.fixture
def both_headers() -> dict[str, str]:
    """Both headers — use when an endpoint accepts either."""
    return {"X-Api-Key": API_KEY, "X-Elevated-Key": ELEVATED_KEY}


# ── Task seeding helper ────────────────────────────────────────────────────────

async def seed_task(
    pool: asyncpg.Pool,
    *,
    intent: str,
    concept_id: str,
    concept_version: str,
    condition_id: str,
    condition_version: str,
    action_id: str,
    action_version: str,
    entity_scope: str = "all",
    delivery: dict | None = None,
    context_version: str | None = None,
    guardrails_version: str | None = None,
) -> str:
    """
    Insert a task row directly, bypassing POST /tasks (which requires LLM).

    Matches the schema written by TaskStore.create(). Returns the auto-generated
    task_id assigned by the DB.

    Workaround: POST /tasks calls TaskAuthoringService which invokes an LLM.
    E2E tests cannot call an LLM, so they seed the task row directly using
    the exact column set that TaskStore.create() writes.
    """
    import json as _json
    if delivery is None:
        delivery = {"type": "webhook", "endpoint": "https://example.com/e2e-webhook"}
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO tasks (
                intent,
                concept_id, concept_version,
                condition_id, condition_version,
                action_id, action_version,
                entity_scope, delivery, status,
                context_version, guardrails_version
            ) VALUES (
                $1,
                $2, $3,
                $4, $5,
                $6, $7,
                $8, $9, 'active',
                $10, $11
            )
            RETURNING task_id
            """,
            intent,
            concept_id, concept_version,
            condition_id, condition_version,
            action_id, action_version,
            entity_scope, _json.dumps(delivery),
            context_version, guardrails_version,
        )
        return row["task_id"]


# ── MockLLMClient injection fixtures ───────────────────────────────────────────
#
# These fixtures extend the standard e2e_client by overriding the
# get_task_authoring_service FastAPI dependency so that the TaskAuthoringService
# receives a MockLLMClient instead of the real LLM client.
#
# Injection approach: FastAPI app.dependency_overrides replaces the dependency
# at the app level without modifying any production code.  The override
# function rebuilds TaskAuthoringService with the same stores and guardrails
# that the original dependency would have used, but substitutes the mock LLM.
#
# The optional ``guardrails`` parameter lets individual tests inject specific
# guardrails configurations directly (bypassing the guardrails_store lookup)
# to test guardrails-dependent behaviour without requiring a live
# GuardrailsStore or POST /guardrails HTTP calls.

from app.api.routes.execute import get_execute_service
from app.api.routes.tasks import get_task_authoring_service
from app.persistence.db import get_db
from app.registry.definitions import DefinitionRegistry
from app.runtime.data_resolver import DataResolver
from app.services.execute import ExecuteService
from app.services.task_authoring import TaskAuthoringService
from app.stores import ContextStore, DefinitionStore, TaskStore
from tests.mocks.mock_connector import MockTableConnector
from tests.mocks.mock_llm_client import MockLLMClient


def _make_mock_override(
    mock_llm: MockLLMClient,
    guardrails=None,
):
    """
    Return a FastAPI dependency override that wires MockLLMClient into
    TaskAuthoringService.

    Parameters
    ──────────
    mock_llm:   The MockLLMClient instance to inject.
    guardrails: Optional Guardrails object to inject directly (bypasses the
                guardrails_store lookup).  When None, reads from app.state
                as the normal dependency would.
    """
    import asyncpg
    from fastapi import Depends, Request

    async def _override(
        request: Request,
        pool: asyncpg.Pool = Depends(get_db),
    ) -> TaskAuthoringService:
        task_store = TaskStore(pool)
        definition_registry = DefinitionRegistry(store=DefinitionStore(pool))
        guardrails_store = getattr(request.app.state, "guardrails_store", None)

        # Use caller-supplied guardrails when provided; otherwise fall back to
        # the store (which may be None in the test app → no guardrails).
        if guardrails is not None:
            effective_guardrails = guardrails
        else:
            effective_guardrails = (
                guardrails_store.get_guardrails()
                if guardrails_store is not None and guardrails_store.is_loaded()
                else None
            )

        return TaskAuthoringService(
            task_store=task_store,
            definition_registry=definition_registry,
            guardrails=effective_guardrails,
            context_store=ContextStore(pool),
            guardrails_store=guardrails_store,
            llm_client=mock_llm,
        )

    return _override


@pytest.fixture
def mock_llm_e2e_client(e2e_setup):
    """
    Function-scoped fixture — yields (client, pool, run_db, mock_llm).

    Identical to ``e2e_client`` but injects MockLLMClient into
    TaskAuthoringService so that POST /tasks is exercisable without a live LLM.

    client   — Starlette TestClient with full HTTP stack
    pool     — asyncpg pool for direct DB access (seeding/querying)
    run_db   — helper: run_db(coro) executes async SQL in the fixture loop
    mock_llm — the MockLLMClient instance (exposes call_count, last_intent, …)
    """
    pool, run_db = e2e_setup
    mock_llm = MockLLMClient()
    app = _make_e2e_app()
    app.dependency_overrides[get_task_authoring_service] = _make_mock_override(mock_llm)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, pool, run_db, mock_llm
    except RuntimeError as exc:
        if "Cannot connect" in str(exc):
            pytest.skip(f"Test database unavailable: {exc}")
            return
        raise


def make_mock_llm_app(guardrails=None) -> tuple[Any, MockLLMClient]:
    """
    Build a FastAPI test app with MockLLMClient injected, plus optional
    direct guardrails injection.

    Returns (app, mock_llm).  Callers wrap this in a TestClient.
    Used by tests that need custom guardrails (Tests 4, 5).
    """
    mock_llm = MockLLMClient()
    app = _make_e2e_app()
    app.dependency_overrides[get_task_authoring_service] = _make_mock_override(
        mock_llm, guardrails=guardrails
    )
    return app, mock_llm


@pytest.fixture
def mock_llm_e2e_client_with_guardrails(e2e_setup):
    """
    Factory fixture — yields a callable that produces (client, pool, run_db, mock_llm)
    with caller-supplied Guardrails injected into TaskAuthoringService.

    Usage in test::

        def test_foo(mock_llm_e2e_client_with_guardrails, api_headers):
            make_client = mock_llm_e2e_client_with_guardrails
            guardrails = Guardrails(...)
            with make_client(guardrails) as (client, pool, run_db, mock_llm):
                r = client.post("/tasks", json={...}, headers=api_headers)
    """
    from contextlib import contextmanager

    pool, run_db = e2e_setup

    @contextmanager
    def _make(guardrails=None):
        mock_llm = MockLLMClient()
        app = _make_e2e_app()
        app.dependency_overrides[get_task_authoring_service] = _make_mock_override(
            mock_llm, guardrails=guardrails
        )
        try:
            with TestClient(app, raise_server_exceptions=True) as client:
                yield client, pool, run_db, mock_llm
        except RuntimeError as exc:
            if "Cannot connect" in str(exc):
                pytest.skip(f"Test database unavailable: {exc}")
                return
            raise

    yield _make


# ── MockConnector injection fixtures ───────────────────────────────────────────
#
# These fixtures extend the standard e2e_client by overriding the
# get_execute_service FastAPI dependency so that DataResolver uses
# MockTableConnector for primitive fetching instead of the empty MockConnector
# that the production code uses when no real connectors are configured.
#
# Injection approach: patch _make_resolver on the ExecuteService instance so
# that every DataResolver it creates uses our table-backed connector.  This
# exercises the real evaluation pipeline (DAGBuilder, ConceptExecutor,
# strategy evaluation) with controlled primitive values — the production data
# path short of a real database or REST endpoint.
#
# The optional ``async_connector`` parameter lets tests inject a
# MockAsyncRestConnector into the async_connector_registry path with
# caller-supplied primitive_sources, testing the full async fetch flow.


def _make_execute_service_override(
    mock_connector: MockTableConnector | None = None,
    async_connector: Any = None,
    primitive_sources: dict | None = None,
) -> Any:
    """
    Return a FastAPI dependency override that injects mock connectors into
    ExecuteService._make_resolver().

    Parameters
    ──────────
    mock_connector:    Sync MockTableConnector used as the default connector
                       for all primitives not served by async_connector.
    async_connector:   Optional MockAsyncRestConnector (or similar async object).
                       When provided, it is registered in the async_connector_registry
                       under async_connector.connector_name.
    primitive_sources: Required when async_connector is set — maps primitive names
                       to PrimitiveSourceConfig so DataResolver routes them through
                       the async connector.
    """
    import asyncpg
    from fastapi import Depends, Request

    from app.models.config import PrimitiveSourceConfig

    effective_sync = mock_connector or MockTableConnector()

    async def _override(
        request: Request,
        pool: asyncpg.Pool = Depends(get_db),
    ) -> ExecuteService:
        service = ExecuteService(pool=pool)

        # Build async registry if an async connector was provided.
        async_registry: dict = {}
        effective_sources: dict = {}
        if async_connector is not None:
            name = getattr(async_connector, "connector_name", "async_mock")
            async_registry[name] = async_connector
            effective_sources = primitive_sources or {}

        def _patched_make_resolver() -> DataResolver:
            return DataResolver(
                connector=effective_sync,
                backoff_base=0.0,
                primitive_sources=effective_sources,
                async_connector_registry=async_registry,
            )

        service._make_resolver = _patched_make_resolver  # type: ignore[method-assign]
        return service

    return _override


@pytest.fixture
def mock_connector_e2e_client(e2e_setup):
    """
    Factory fixture — yields a callable that produces (client, pool, run_db)
    with MockTableConnector injected into the execution pipeline.

    Identical to ``e2e_client`` except the primitive data connector returns
    values from a caller-supplied table instead of always returning None.

    Usage::

        def test_foo(mock_connector_e2e_client, api_headers, elevated_headers):
            data = {"account.active_user_rate_30d": {"account_001": 0.25}}
            connector = MockTableConnector(data)
            with mock_connector_e2e_client(connector) as (client, pool, run_db):
                r = client.post("/evaluate/full", json={...}, headers=api_headers)

    Async REST connector variant::

        connector = MockAsyncRestConnector(data, connector_name="rest_mock")
        sources = {"account.active_user_rate_30d": PrimitiveSourceConfig(
            connector="rest_mock", query="/api/v1/{entity_id}")}
        with mock_connector_e2e_client(
            async_connector=connector, primitive_sources=sources
        ) as (client, pool, run_db):
            ...
    """
    from contextlib import contextmanager

    pool, run_db = e2e_setup

    @contextmanager
    def _make(
        mock_connector: MockTableConnector | None = None,
        async_connector: Any = None,
        primitive_sources: dict | None = None,
    ):
        app = _make_e2e_app()
        app.dependency_overrides[get_execute_service] = _make_execute_service_override(
            mock_connector=mock_connector,
            async_connector=async_connector,
            primitive_sources=primitive_sources,
        )
        try:
            with TestClient(app, raise_server_exceptions=True) as client:
                yield client, pool, run_db
        except RuntimeError as exc:
            if "Cannot connect" in str(exc):
                pytest.skip(f"Test database unavailable: {exc}")
                return
            raise

    yield _make


# ── MockWebhookServer injection fixtures ───────────────────────────────────────
#
# These fixtures patch httpx.AsyncClient in app.runtime.action_trigger so that
# all outbound webhook HTTP calls are intercepted and recorded by MockWebhookServer
# without making real network requests.
#
# mock_webhook_fixture:
#   Standalone — just patches httpx and yields the server.  Use alongside
#   mock_connector_e2e_client when you need both DB/connector + webhook mocking.
#
# mock_webhook_connector_e2e_client:
#   Combined factory fixture — yields (client, pool, run_db, webhook).
#   Combines MockTableConnector injection with MockWebhookServer patching.

from tests.mocks.mock_webhook_server import MockWebhookServer


@pytest.fixture
def mock_webhook_fixture():
    """
    Function-scoped fixture — yields a running MockWebhookServer.

    Patches ``app.runtime.action_trigger.httpx.AsyncClient`` for the duration
    of the test so all outbound webhook HTTP calls are intercepted and recorded.
    No real network calls are made.

    Usage::

        def test_foo(mock_connector_e2e_client, mock_webhook_fixture, api_headers):
            webhook = mock_webhook_fixture
            data = {"account.rate": {"acct_1": 0.25}}
            connector = MockTableConnector(data)
            with mock_connector_e2e_client(connector) as (client, pool, run_db):
                ...  # register definitions, call /evaluate/full
                assert webhook.call_count == 1
    """
    server = MockWebhookServer()
    with server:
        yield server


@pytest.fixture
def mock_webhook_connector_e2e_client(e2e_setup):
    """
    Factory fixture — yields a callable producing (client, pool, run_db, webhook).

    Combines MockTableConnector injection into ExecuteService with
    MockWebhookServer patching of httpx.AsyncClient.  The webhook server is
    active for the full duration of the ``with`` block.

    Usage::

        def test_foo(mock_webhook_connector_e2e_client, api_headers, elevated_headers):
            data = {"account.rate": {"acct_1": 0.25}}
            connector = MockTableConnector(data)
            with mock_webhook_connector_e2e_client(connector) as (client, pool, run_db, webhook):
                ...  # register definitions, call /evaluate/full
                assert webhook.call_count == 1

    Configuring the webhook response (e.g. failure simulation)::

        webhook = MockWebhookServer(status_code=500)
        with mock_webhook_connector_e2e_client(connector, webhook) as (..., webhook):
            ...
    """
    from contextlib import contextmanager

    pool, run_db = e2e_setup

    @contextmanager
    def _make(
        mock_connector: MockTableConnector | None = None,
        mock_webhook: MockWebhookServer | None = None,
    ):
        effective_connector = mock_connector or MockTableConnector()
        effective_webhook = mock_webhook or MockWebhookServer()

        app = _make_e2e_app()
        app.dependency_overrides[get_execute_service] = _make_execute_service_override(
            mock_connector=effective_connector,
        )
        with effective_webhook:
            try:
                with TestClient(app, raise_server_exceptions=True) as client:
                    yield client, pool, run_db, effective_webhook
            except RuntimeError as exc:
                if "Cannot connect" in str(exc):
                    pytest.skip(f"Test database unavailable: {exc}")
                    return
                raise

    yield _make
