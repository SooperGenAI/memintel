"""
tests/unit/test_direct_task.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for POST /tasks/direct — no LLM, no DB, no real stores.

Coverage
────────
  test_direct_task_creation         happy path; threshold strategy → Task 200
  test_direct_task_concept_not_found  concept absent in registry → 404
  test_direct_task_idempotent       existing task found → returned, create() not called
  test_direct_task_all_strategies   each of the 6 strategy types passes validation

Design
──────
The FastAPI app is built once per test using _make_app(), which overrides
_get_direct_stores with mock (TaskStore, DefinitionStore) tuples.  No shared
mutable state between tests.  Uses FastAPI TestClient (ASGI, no real network).
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

# aioredis uses `distutils` removed in Python 3.12+; stub before app imports.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import tasks as tasks_route
from app.api.routes.tasks import _get_direct_stores
from app.models.errors import MemintelError, memintel_error_handler
from app.models.task import DeliveryConfig, DeliveryType, Task, TaskStatus


# ── Shared test constants ──────────────────────────────────────────────────────

_CONCEPT_ID   = "customer.days_sales_outstanding"
_CONCEPT_V    = "1.0"
_CONDITION_ID = "dso_above_threshold"
_ACTION_ID    = "dso_alert"
_TASK_ID      = "task-direct-001"

_CONCEPT_BODY = {"concept_id": _CONCEPT_ID, "version": _CONCEPT_V}

_DELIVERY_JSON = {"type": "notification", "channel": "default"}

_BASE_REQUEST = {
    "intent": "Monitor DSO above threshold",
    "entity_scope": "customer",
    "concept_id": _CONCEPT_ID,
    "concept_version": _CONCEPT_V,
    "condition": {
        "id": _CONDITION_ID,
        "strategy": "threshold",
        "params": {"value": 45, "direction": "above"},
    },
    "action": {
        "id": _ACTION_ID,
        "type": "notification",
        "channel": "default",
    },
    "delivery": _DELIVERY_JSON,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task() -> Task:
    return Task(
        task_id=_TASK_ID,
        intent="Monitor DSO above threshold",
        concept_id=_CONCEPT_ID,
        concept_version=_CONCEPT_V,
        condition_id=_CONDITION_ID,
        condition_version="1.0",
        action_id=_ACTION_ID,
        action_version="1.0",
        entity_scope="customer",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="default"),
        status=TaskStatus.ACTIVE,
    )


def _make_app(task_store: object, def_store: object) -> FastAPI:
    @asynccontextmanager
    async def _null_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=_null_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(tasks_route.router)

    app.dependency_overrides[_get_direct_stores] = lambda: (task_store, def_store)
    return app


def _def_store(concept_body=_CONCEPT_BODY) -> MagicMock:
    store = MagicMock()
    store.get = AsyncMock(return_value=concept_body)
    store.register = AsyncMock(return_value=MagicMock())
    return store


def _task_store(existing: list | None = None, created: Task | None = None) -> MagicMock:
    store = MagicMock()
    store.find_by_condition_version = AsyncMock(return_value=existing or [])
    store.create = AsyncMock(return_value=created or _make_task())
    return store


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_direct_task_creation() -> None:
    """Happy path: registers condition + action defs and creates a Task."""
    ts = _task_store()
    ds = _def_store()
    app = _make_app(ts, ds)

    with TestClient(app) as client:
        resp = client.post("/tasks/direct", json=_BASE_REQUEST)

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == _TASK_ID
    assert data["concept_id"] == _CONCEPT_ID
    assert data["condition_id"] == _CONDITION_ID
    assert data["condition_version"] == "1.0"
    assert data["action_id"] == _ACTION_ID
    assert data["action_version"] == "1.0"
    assert data["entity_scope"] == "customer"

    # concept lookup, then condition register, then action register
    assert ds.get.call_count == 1
    assert ds.register.call_count == 2
    ts.create.assert_called_once()


def test_direct_task_concept_not_found() -> None:
    """Returns 404 when concept_id + concept_version is absent from the registry."""
    ts = _task_store()
    ds = _def_store(concept_body=None)  # get() returns None → concept missing
    app = _make_app(ts, ds)

    with TestClient(app) as client:
        resp = client.post("/tasks/direct", json=_BASE_REQUEST)

    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["type"] == "not_found"
    ts.create.assert_not_called()


def test_direct_task_idempotent() -> None:
    """
    When a non-deleted task already exists for the given condition_id, the
    existing task is returned and TaskStore.create() is never called.
    """
    existing = _make_task()
    ts = _task_store(existing=[existing])
    ds = _def_store()
    app = _make_app(ts, ds)

    with TestClient(app) as client:
        resp = client.post("/tasks/direct", json=_BASE_REQUEST)

    assert resp.status_code == 200
    assert resp.json()["task_id"] == _TASK_ID
    ts.create.assert_not_called()


def test_direct_task_all_strategies() -> None:
    """Each of the six strategy types is accepted and produces HTTP 200."""
    strategy_cases: list[tuple[str, dict]] = [
        ("threshold",  {"value": 45.0, "direction": "above"}),
        ("percentile", {"value": 90.0, "direction": "top"}),
        ("z_score",    {"threshold": 2.0, "direction": "above"}),
        ("change",     {"value": 0.15, "direction": "increase", "window": "30d"}),
        ("equals",     {"value": "overdue"}),
        ("composite",  {
            "operator": "AND",
            "operands": [
                {"condition_id": "cond.dso_high", "condition_version": "1.0"},
                {"condition_id": "cond.dso_rising", "condition_version": "1.0"},
            ],
        }),
    ]

    for strategy_type, params in strategy_cases:
        ts = _task_store()
        ds = _def_store()
        app = _make_app(ts, ds)

        body = {
            **_BASE_REQUEST,
            "condition": {
                "id": f"cond_{strategy_type}",
                "strategy": strategy_type,
                "params": params,
            },
        }

        with TestClient(app) as client:
            resp = client.post("/tasks/direct", json=body)

        assert resp.status_code == 200, (
            f"Strategy '{strategy_type}' returned {resp.status_code}: {resp.text}"
        )
