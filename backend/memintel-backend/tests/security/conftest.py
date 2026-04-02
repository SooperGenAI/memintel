"""
tests/security/conftest.py
──────────────────────────────────────────────────────────────────────────────
Security test infrastructure — same real-DB TestClient as contract tests,
augmented with auth-variant fixtures and namespace isolation helpers.

Every security test gets a fresh TestClient (and fresh DB state) via the
function-scoped `app_client` fixture.

Database: postgresql://postgres:admin@localhost:5433/memintel_test
Tests are skipped gracefully if the database is unavailable.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

# Stub aioredis before any app import — prevents ImportError in test context.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import asyncpg
import pytest
from fastapi import FastAPI, HTTPException, Request
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

ELEVATED_KEY = "security-test-elevated-key"
API_KEY = "security-test-api-key"
WRONG_ELEVATED_KEY = "wrong-elevated-key-x"
WRONG_API_KEY = "wrong-api-key-x"

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


async def _create_test_database() -> None:
    conn = await asyncpg.connect(_admin_url())
    try:
        await conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = 'memintel_test' AND pid <> pg_backend_pid()
            """
        )
        await conn.execute("DROP DATABASE IF EXISTS memintel_test")
        await conn.execute("CREATE DATABASE memintel_test")
    finally:
        await conn.close()


async def _truncate_all(pool: asyncpg.Pool) -> None:
    table_list = ", ".join(_ALL_TABLES)
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table_list} RESTART IDENTITY CASCADE")


# ── Session-level DB setup ─────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _database_setup():
    """
    Create and migrate the memintel_test database once per pytest session.

    Skips all security tests gracefully if PostgreSQL is unreachable or
    migrations fail.
    """
    setup_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(setup_loop)
    try:
        setup_loop.run_until_complete(_create_test_database())
    except Exception as exc:
        setup_loop.close()
        asyncio.set_event_loop(None)
        pytest.skip(
            f"Cannot create test database — is PostgreSQL running on "
            f"{_admin_url()}? Error: {exc}"
        )
        return
    finally:
        if not setup_loop.is_closed():
            setup_loop.close()

    asyncio.set_event_loop(None)

    try:
        _run_alembic_migrations()
    except Exception as exc:
        pytest.skip(f"Alembic migrations failed: {exc}")
        return

    yield  # all security tests run here


# ── HTTP exception handler (mirrors main.py) ───────────────────────────────────

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


# ── Test app factory ───────────────────────────────────────────────────────────

def _make_test_app() -> FastAPI:
    """
    Build a FastAPI app with all production routers.

    The asyncpg pool is created INSIDE the lifespan so it is bound to the
    TestClient's anyio event loop — no cross-loop errors.
    """
    @asynccontextmanager
    async def _test_lifespan(app: FastAPI):
        try:
            pool = await asyncpg.create_pool(
                TEST_DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to test database: {exc}") from exc

        await _truncate_all(pool)

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

    app = FastAPI(title="Memintel Backend API", lifespan=_test_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.add_exception_handler(HTTPException, _http_exc_handler)

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


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def app_client(_database_setup):
    """
    Function-scoped fixture — yields (TestClient, FastAPI app).

    Each test gets a fresh TestClient and a clean (truncated) database.
    Skips if the database is unavailable.
    """
    app = _make_test_app()
    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, app
    except RuntimeError as exc:
        if "Cannot connect" in str(exc):
            pytest.skip(f"Test database unavailable: {exc}")
            return
        raise


@pytest.fixture
def valid_elevated_headers() -> dict[str, str]:
    """HTTP headers with the correct elevated key."""
    return {"X-Elevated-Key": ELEVATED_KEY}


@pytest.fixture
def valid_api_key_headers() -> dict[str, str]:
    """HTTP headers with the correct API key."""
    return {"X-Api-Key": API_KEY}


@pytest.fixture
def wrong_elevated_headers() -> dict[str, str]:
    """HTTP headers with a wrong elevated key — expect 403."""
    return {"X-Elevated-Key": WRONG_ELEVATED_KEY}


@pytest.fixture
def wrong_api_key_headers() -> dict[str, str]:
    """HTTP headers with a wrong API key — expect 401."""
    return {"X-Api-Key": WRONG_API_KEY}


@pytest.fixture
def no_headers() -> dict[str, str]:
    """Empty headers dict — no authentication provided."""
    return {}


@pytest.fixture
def org_a_headers() -> dict[str, str]:
    """Headers representing an org-a client (API key only, no elevated)."""
    return {"X-Api-Key": API_KEY}


@pytest.fixture
def org_b_headers() -> dict[str, str]:
    """
    Headers representing an org-b client.

    SECURITY GAP: The system has a single global API key — there is no
    per-tenant key. Both org-a and org-b use the same API_KEY. Namespace
    isolation is therefore purely by convention (namespace filter), not
    enforced by authentication.
    """
    return {"X-Api-Key": API_KEY}


@pytest.fixture
def permissive_client(_database_setup):
    """
    Function-scoped TestClient with raise_server_exceptions=False.

    Used for edge-case input-sanitisation tests where the server may raise
    an unhandled exception (which propagates to the test with the default
    raise_server_exceptions=True).  With raise_server_exceptions=False,
    unhandled exceptions are returned as HTTP 500 responses instead.
    """
    app = _make_test_app()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    except RuntimeError as exc:
        if "Cannot connect" in str(exc):
            pytest.skip(f"Test database unavailable: {exc}")
            return
        raise
