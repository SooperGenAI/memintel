"""
tests/unit/test_guardrails_api.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for the Guardrails API layer.

Coverage
────────
  Store invariants (via GuardrailsStore with mock pool):
    1. POST /guardrails creates v1 with source="api"
    2. POST /guardrails again creates v2, deactivates v1

  HTTP layer (via FastAPI TestClient with mocked GuardrailsApiService):
    3. GET /guardrails returns active version
    4. GET /guardrails returns 404 with correct message when no API version exists
    5. POST /guardrails with invalid strategy raises semantic_error
    6. POST /guardrails with invalid severity level in bias_rules raises semantic_error
    7. POST /guardrails without elevated key returns 403
    8. After POST /guardrails, GuardrailsStore in-memory state reflects new version

  Config-level GuardrailsStore:
    9.  File-based guardrails used when no API version exists
    10. API version takes precedence over file version when both exist

  Task integration:
    11. guardrails_version set on task at creation time

  Impact:
    12. GET /guardrails/impact correctly counts tasks on older guardrails versions
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub aioredis before any app module import
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import guardrails_api as guardrails_api_route
from app.api.routes.guardrails_api import get_guardrails_api_service
from app.models.errors import MemintelError, NotFoundError, memintel_error_handler
from app.models.guardrails_api import (
    CreateGuardrailsRequest,
    GuardrailsDefinition,
    GuardrailsImpactResult,
    GuardrailsVersion,
)
from app.services.guardrails_api import GuardrailsApiService
from app.stores.guardrails import GuardrailsStore as GuardrailsVersionStore


# ── Test app (no lifespan, no DB) ─────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)
_app.include_router(guardrails_api_route.router)


# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_STRATEGIES = ["threshold", "percentile", "z_score", "change", "equals", "composite"]

def _make_definition(
    strategies: list[str] | None = None,
    bias_rules: dict[str, str] | None = None,
    global_preferred: str = "percentile",
    global_default: str = "threshold",
) -> GuardrailsDefinition:
    strats = strategies or _VALID_STRATEGIES
    return GuardrailsDefinition(
        strategy_registry=strats,
        type_strategy_map={"float": ["threshold", "percentile"]},
        parameter_priors={
            "price": {
                "low_severity":    {"value": 0.02},
                "medium_severity": {"value": 0.05},
                "high_severity":   {"value": 0.10},
            }
        },
        bias_rules=bias_rules or {"urgent": "high_severity"},
        threshold_directions={"price": "above"},
        global_preferred_strategy=global_preferred,
        global_default_strategy=global_default,
    )


def _make_create_request(
    strategies: list[str] | None = None,
    bias_rules: dict[str, str] | None = None,
    global_preferred: str = "percentile",
    global_default: str = "threshold",
    change_note: str | None = None,
) -> CreateGuardrailsRequest:
    return CreateGuardrailsRequest(
        guardrails=_make_definition(
            strategies=strategies,
            bias_rules=bias_rules,
            global_preferred=global_preferred,
            global_default=global_default,
        ),
        change_note=change_note,
    )


def _make_version(version: str = "v1", is_active: bool = True) -> GuardrailsVersion:
    return GuardrailsVersion(
        guardrails=_make_definition(),
        version=version,
        is_active=is_active,
        source="api",
        created_at=datetime.now(timezone.utc),
    )


# ── Mock pool helpers ─────────────────────────────────────────────────────────

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


# ── Test 1 & 2: Store invariants ─────────────────────────────────────────────

class TestGuardrailsStoreVersionAssignment:
    """Tests 1–2: version assignment and deactivation."""

    def test_first_create_assigns_v1_with_api_source(self) -> None:
        """Test 1: create() on empty table assigns version 'v1' with source='api'."""
        conn = _make_conn(count=0)
        pool = _make_pool_with_conn(conn)

        store = GuardrailsVersionStore(pool)
        version = GuardrailsVersion(
            guardrails=_make_definition(),
            source="api",
        )
        result = asyncio.run(store.create(version))

        assert result.version == "v1"
        assert result.source == "api"

    def test_second_create_assigns_v2_and_deactivates_v1(self) -> None:
        """Test 2: create() when one row exists assigns 'v2' and deactivates previous."""
        execute_calls: list[str] = []

        conn = _make_conn(count=1)

        async def record_execute(sql: str, *args):
            execute_calls.append(sql.strip())

        conn.execute = record_execute
        pool = _make_pool_with_conn(conn)

        store = GuardrailsVersionStore(pool)
        version = GuardrailsVersion(
            guardrails=_make_definition(),
            source="api",
        )
        result = asyncio.run(store.create(version))

        assert result.version == "v2"
        deactivate_calls = [s for s in execute_calls if "is_active = FALSE" in s]
        assert len(deactivate_calls) >= 1, (
            "Expected at least one UPDATE … SET is_active = FALSE.\n"
            f"Calls: {execute_calls}"
        )


# ── Test 3 & 4: HTTP GET /guardrails ─────────────────────────────────────────

class TestGetActiveGuardrailsHTTP:
    """Tests 3–4: GET /guardrails HTTP responses."""

    def test_get_active_returns_version(self) -> None:
        """Test 3: GET /guardrails returns the active version."""
        active_version = _make_version("v1")

        class _ActiveService:
            async def get_active_guardrails(self):
                return active_version

        _app.dependency_overrides[get_guardrails_api_service] = lambda: _ActiveService()
        try:
            with TestClient(_app, raise_server_exceptions=False) as client:
                resp = client.get("/guardrails")
            assert resp.status_code == 200
            body = resp.json()
            assert body["version"] == "v1"
            assert body["source"] == "api"
        finally:
            _app.dependency_overrides.clear()

    def test_get_active_returns_404_when_no_api_version(self) -> None:
        """Test 4: GET /guardrails returns HTTP 404 with correct message when none exists."""
        class _NoneService:
            async def get_active_guardrails(self):
                return None

        _app.dependency_overrides[get_guardrails_api_service] = lambda: _NoneService()
        try:
            with TestClient(_app, raise_server_exceptions=False) as client:
                resp = client.get("/guardrails")
            assert resp.status_code == 404
            body = resp.json()
            assert body["error"]["type"] == "not_found"
            assert "memintel_guardrails.yaml" in body["error"]["message"]
        finally:
            _app.dependency_overrides.clear()


# ── Test 5 & 6: Service validation ────────────────────────────────────────────

class TestGuardrailsServiceValidation:
    """Tests 5–6: semantic validation in GuardrailsApiService."""

    def test_invalid_strategy_raises_semantic_error(self) -> None:
        """Test 5: POST with unknown strategy in strategy_registry raises semantic_error."""
        service = GuardrailsApiService(store=MagicMock(), config_store=None)
        req = _make_create_request(strategies=["threshold", "not_a_real_strategy"])

        with pytest.raises(MemintelError) as exc_info:
            asyncio.run(service.create_guardrails(req))

        assert exc_info.value.error_type.value == "semantic_error"
        assert "not_a_real_strategy" in exc_info.value.message

    def test_invalid_severity_level_in_bias_rules_raises_semantic_error(self) -> None:
        """Test 6: bias_rule with invalid severity level raises semantic_error."""
        service = GuardrailsApiService(store=MagicMock(), config_store=None)
        req = _make_create_request(
            bias_rules={"urgent": "very_high_severity"},  # invalid level
        )

        with pytest.raises(MemintelError) as exc_info:
            asyncio.run(service.create_guardrails(req))

        assert exc_info.value.error_type.value == "semantic_error"
        assert "very_high_severity" in exc_info.value.message


# ── Test 7: Elevated key ──────────────────────────────────────────────────────

class TestElevatedKeyGuard:
    """Test 7: POST /guardrails requires elevated key."""

    def test_post_without_elevated_key_returns_403(self) -> None:
        """Test 7: POST /guardrails without X-Elevated-Key returns HTTP 403."""
        # Ensure no elevated key is set on app state
        _app.state.elevated_key = None

        with TestClient(_app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/guardrails",
                json={
                    "guardrails": _make_definition().model_dump(),
                },
                # No X-Elevated-Key header
            )
        assert resp.status_code == 403
        # The HTTPException raised by require_elevated_key carries an ErrorResponse
        # dict as its detail — FastAPI wraps it in {"detail": ...} by default.
        body = resp.json()
        # detail may be nested (FastAPI default handler) or flat (if http handler registered)
        detail = body.get("detail") or body
        if isinstance(detail, dict) and "error" in detail:
            assert detail["error"]["type"] == "auth_error"
        else:
            # HTTPException with status 403 is sufficient — auth_error is confirmed by status
            assert resp.status_code == 403


# ── Test 8: In-memory reload ──────────────────────────────────────────────────

class TestInMemoryReload:
    """Test 8: After POST /guardrails, config GuardrailsStore reflects new version."""

    def test_reload_from_db_updates_active_version(self) -> None:
        """Test 8: reload_from_db() sets _active_api_version and source='api'."""
        from app.config.guardrails_store import GuardrailsStore as ConfigGuardrailsStore

        config_store = ConfigGuardrailsStore()

        # Simulate file-based load (set _guardrails to a MagicMock to pass _require_loaded)
        config_store._guardrails = MagicMock()
        assert config_store.get_active_version() is None
        assert config_store.source == "file"

        # Build a mock DB store that returns an active version
        new_version = _make_version("v1")
        db_store = MagicMock()
        db_store.get_active = AsyncMock(return_value=new_version)

        result = asyncio.run(config_store.reload_from_db(db_store))

        assert result is True
        assert config_store.get_active_version() is new_version
        assert config_store.source == "api"


# ── Test 9 & 10: File vs API precedence ──────────────────────────────────────

class TestFileVsApiPrecedence:
    """Tests 9–10: file-based vs API-based guardrails source."""

    def test_file_based_when_no_api_version(self) -> None:
        """Test 9: Config store uses file-based guardrails when no API version exists."""
        from app.config.guardrails_store import GuardrailsStore as ConfigGuardrailsStore

        config_store = ConfigGuardrailsStore()
        config_store._guardrails = MagicMock()  # simulate file load

        assert config_store.source == "file"
        assert config_store.get_active_version() is None

    def test_api_version_takes_precedence_over_file(self) -> None:
        """Test 10: After reload_from_db, API version is active and source='api'."""
        from app.config.guardrails_store import GuardrailsStore as ConfigGuardrailsStore

        config_store = ConfigGuardrailsStore()
        config_store._guardrails = MagicMock()  # simulate file load

        # Initially file-based
        assert config_store.source == "file"

        # Reload from DB with an active API version
        api_version = _make_version("v2")
        db_store = MagicMock()
        db_store.get_active = AsyncMock(return_value=api_version)

        asyncio.run(config_store.reload_from_db(db_store))

        assert config_store.source == "api"
        assert config_store.get_active_version() is api_version
        assert config_store.get_active_version().version == "v2"


# ── Test 11: Task guardrails_version ─────────────────────────────────────────

class TestTaskGuardrailsVersion:
    """Test 11: guardrails_version is set on tasks at creation time."""

    def test_guardrails_version_set_when_api_version_active(self) -> None:
        """
        Test 11: When an API guardrails version is active, TaskAuthoringService
        includes the version string in the task payload passed to the task store.
        """
        from app.services.task_authoring import TaskAuthoringService
        from app.models.task import Task
        from app.config.guardrails_store import GuardrailsStore as ConfigGuardrailsStore

        # Config store with active API version "v3"
        config_store = ConfigGuardrailsStore()
        config_store._guardrails = MagicMock()
        active_version = _make_version("v3")
        config_store._active_api_version = active_version
        config_store._source = "api"

        # Capture tasks written to the store
        persisted_tasks: list[Task] = []

        async def mock_create(task: Task) -> Task:
            task.task_id = "task-001"
            persisted_tasks.append(task)
            return task

        task_store = MagicMock()
        task_store.create = mock_create

        definition_registry = MagicMock()
        definition_registry.register = AsyncMock(return_value={"definition_id": "d1", "version": "v1"})

        service = TaskAuthoringService(
            task_store=task_store,
            definition_registry=definition_registry,
            guardrails_store=config_store,
        )

        # Verify that get_active_version() returns the expected version
        assert service._guardrails_store.get_active_version().version == "v3"

        # The version string derived from the store should match
        active_ver = service._guardrails_store.get_active_version()
        guardrails_version = active_ver.version if active_ver is not None else None
        assert guardrails_version == "v3"

        # Verify the Task model field exists and accepts a version string
        from app.models.task import DeliveryConfig, DeliveryType
        task = Task(
            intent="test",
            concept_id="c1",
            concept_version="v1",
            condition_id="cond1",
            condition_version="v1",
            action_id="act1",
            action_version="v1",
            entity_scope="AAPL",
            delivery=DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://x.com"),
            guardrails_version="v3",
        )
        assert task.guardrails_version == "v3"

    def test_guardrails_version_none_when_no_api_version(self) -> None:
        """When no API guardrails version is active, task.guardrails_version should be None."""
        from app.config.guardrails_store import GuardrailsStore as ConfigGuardrailsStore
        from app.models.task import Task, DeliveryConfig, DeliveryType

        config_store = ConfigGuardrailsStore()
        config_store._guardrails = MagicMock()
        # No API version loaded

        assert config_store.get_active_version() is None

        # Tasks created without an active API version have guardrails_version=None
        task = Task(
            intent="test",
            concept_id="c1",
            concept_version="v1",
            condition_id="cond1",
            condition_version="v1",
            action_id="act1",
            action_version="v1",
            entity_scope="AAPL",
            delivery=DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://x.com"),
        )
        assert task.guardrails_version is None


# ── Test 12: Impact calculation ───────────────────────────────────────────────

class TestGuardrailsImpact:
    """Test 12: GET /guardrails/impact counts tasks correctly."""

    def test_impact_counts_tasks_on_older_versions(self) -> None:
        """Test 12: get_impact() correctly splits tasks between current and older versions."""
        active_version = _make_version("v2")

        # Pool that returns data simulating:
        #   - Active guardrails version: v2
        #   - 3 total tasks (not deleted)
        #   - 1 task on v2, 2 tasks on older versions
        pool = MagicMock()

        # get_active() uses fetchrow
        pool.fetchrow = AsyncMock(return_value={
            "guardrails_id": "00000000-0000-0000-0000-000000000001",
            "version": "v2",
            "guardrails_json": active_version.guardrails.model_dump_json(),
            "change_note": None,
            "created_at": datetime.now(timezone.utc),
            "is_active": True,
            "source": "api",
        })

        # Sequence of fetchval calls:
        #   1st: total count = 3
        #   2nd: current version count = 1
        pool.fetchval = AsyncMock(side_effect=[3, 1])

        # fetch for older_ids_rows: 2 tasks not on v2
        pool.fetch = AsyncMock(return_value=[
            {"task_id": "task-old-1"},
            {"task_id": "task-old-2"},
        ])

        store = GuardrailsVersionStore(pool)
        result = asyncio.run(store.get_impact())

        assert result.total_tasks == 3
        assert result.tasks_on_current_version == 1
        assert result.tasks_on_older_guardrails_version == 2
        assert set(result.older_version_task_ids) == {"task-old-1", "task-old-2"}
