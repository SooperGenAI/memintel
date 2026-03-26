"""
tests/unit/test_context.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for the Application Context layer.

Coverage
────────
  Store invariants (via ContextStore with a mock pool):
    1. version assignment       — first create → "v1"; second create → "v2"
    2. deactivation             — previous active row is deactivated on create
    3. get_active None          — get_active() returns None when table is empty

  CalibrationBias model validator:
    4. bias_direction recall    — false_negative_cost > false_positive_cost
    5. bias_direction precision — false_positive_cost > false_negative_cost
    6. bias_direction balanced  — equal costs

  HTTP layer (via FastAPI TestClient with mocked ContextService):
    7. GET /context 404         — service raises NotFoundError → HTTP 404
    8. GET /context/impact      — returns ContextImpactResult with all-zero counts
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub aioredis before any app module import (removed in Python 3.12+)
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import context as context_route
from app.api.routes.context import get_context_service
from app.models.context import (
    ApplicationContext,
    BehaviouralContext,
    CalibrationBias,
    ContextImpactResult,
    CreateContextRequest,
    DomainContext,
)
from app.models.errors import NotFoundError, memintel_error_handler, MemintelError
from app.services.context import ContextService
from app.stores.context import ContextStore


# ── Test app (no lifespan, no DB) ─────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)
_app.include_router(context_route.router)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_domain() -> DomainContext:
    return DomainContext(description="Test domain", entities=[], decisions=[])


def _make_context(version: str = "v1", is_active: bool = True) -> ApplicationContext:
    return ApplicationContext(
        domain=_make_domain(),
        version=version,
        is_active=is_active,
    )


def _make_create_request() -> CreateContextRequest:
    return CreateContextRequest(domain=_make_domain())


# ── Store invariants ──────────────────────────────────────────────────────────

def _make_pool_with_conn(conn: AsyncMock) -> MagicMock:
    """Build a mock asyncpg pool whose acquire() context manager yields conn."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=conn),
        __aexit__=AsyncMock(return_value=False),
    ))
    return pool


def _make_conn(count: int = 0) -> AsyncMock:
    """Build a mock asyncpg connection with COUNT(*) returning count."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=count)
    conn.execute = AsyncMock()
    conn.transaction = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))
    return conn


class TestContextStoreVersionAssignment:
    """Tests 1–4: version assignment, deactivation, and get_active None."""

    def test_first_create_assigns_v1(self) -> None:
        """create() on an empty table assigns version 'v1'."""
        conn = _make_conn(count=0)
        pool = _make_pool_with_conn(conn)

        store = ContextStore(pool)
        ctx = ApplicationContext(domain=_make_domain())
        result = asyncio.run(store.create(ctx))

        assert result.version == "v1"

    def test_second_create_assigns_v2(self) -> None:
        """create() when one row already exists assigns version 'v2'."""
        conn = _make_conn(count=1)
        pool = _make_pool_with_conn(conn)

        store = ContextStore(pool)
        ctx = ApplicationContext(domain=_make_domain())
        result = asyncio.run(store.create(ctx))

        assert result.version == "v2"

    def test_create_deactivates_previous_active(self) -> None:
        """create() issues an UPDATE ... SET is_active = FALSE before INSERT."""
        execute_calls: list[str] = []

        conn = _make_conn(count=1)

        async def record_execute(sql: str, *args):
            execute_calls.append(sql.strip())

        conn.execute = record_execute
        pool = _make_pool_with_conn(conn)

        store = ContextStore(pool)
        ctx = ApplicationContext(domain=_make_domain())
        asyncio.run(store.create(ctx))

        deactivate_calls = [s for s in execute_calls if "is_active = FALSE" in s]
        assert len(deactivate_calls) >= 1, (
            "Expected at least one UPDATE … SET is_active = FALSE, got none.\n"
            f"Calls: {execute_calls}"
        )

    def test_get_active_returns_none_when_empty(self) -> None:
        """get_active() returns None when the table has no rows."""
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)

        store = ContextStore(pool)
        result = asyncio.run(store.get_active())

        assert result is None


# ── CalibrationBias model validator ──────────────────────────────────────────

class TestCalibrationBiasDirection:
    """Tests 4–6: bias_direction is always auto-derived."""

    def test_recall_when_fn_cost_higher(self) -> None:
        bias = CalibrationBias(false_negative_cost="high", false_positive_cost="low")
        assert bias.bias_direction == "recall"

    def test_precision_when_fp_cost_higher(self) -> None:
        bias = CalibrationBias(false_negative_cost="low", false_positive_cost="high")
        assert bias.bias_direction == "precision"

    def test_balanced_when_costs_equal(self) -> None:
        bias = CalibrationBias(false_negative_cost="medium", false_positive_cost="medium")
        assert bias.bias_direction == "balanced"

    def test_caller_supplied_bias_direction_is_overwritten(self) -> None:
        """bias_direction supplied by caller must be silently overwritten."""
        bias = CalibrationBias(
            false_negative_cost="high",
            false_positive_cost="low",
            bias_direction="precision",   # caller-supplied wrong value
        )
        assert bias.bias_direction == "recall"


# ── HTTP layer ────────────────────────────────────────────────────────────────

class TestContextHTTP:
    """Tests 7–8: HTTP routes with mocked ContextService."""

    def test_get_active_context_returns_404_when_none(self) -> None:
        """GET /context → HTTP 404 when no active context exists."""
        class _NoneService:
            async def get_active_context(self):
                raise NotFoundError("No active application context exists.")

        _app.dependency_overrides[get_context_service] = lambda: _NoneService()
        try:
            with TestClient(_app, raise_server_exceptions=False) as client:
                resp = client.get("/context")
            assert resp.status_code == 404
            body = resp.json()
            assert body["error"]["type"] == "not_found"
        finally:
            _app.dependency_overrides.clear()

    def test_get_impact_returns_zero_counts(self) -> None:
        """GET /context/impact returns ContextImpactResult with all-zero counts."""
        zero_impact = ContextImpactResult(
            total_tasks=0,
            tasks_on_current_version=0,
            tasks_on_older_versions=0,
            older_version_task_ids=[],
        )

        class _ZeroImpactService:
            async def get_impact(self):
                return zero_impact

        _app.dependency_overrides[get_context_service] = lambda: _ZeroImpactService()
        try:
            with TestClient(_app, raise_server_exceptions=False) as client:
                resp = client.get("/context/impact")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total_tasks"] == 0
            assert body["tasks_on_current_version"] == 0
            assert body["tasks_on_older_versions"] == 0
            assert body["older_version_task_ids"] == []
        finally:
            _app.dependency_overrides.clear()
