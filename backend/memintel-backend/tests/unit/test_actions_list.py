"""
tests/unit/test_actions_list.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for GET /actions list endpoint and DefinitionStore.list_actions().

Coverage:
  1.  GET /actions returns list of ActionDefinitions for the given namespace
  2.  GET /actions returns empty list when no actions exist (not 404)
  3.  GET /actions excludes deprecated actions
  4.  GET /actions without namespace returns HTTP 422
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import actions as actions_route
from app.api.routes.actions import get_definition_store
from app.models.action import (
    ActionDefinition,
    RegisterActionConfig,
    TriggerConfig,
    FireOn,
)
from app.models.errors import MemintelError, memintel_error_handler
from app.models.task import Namespace


# ── Test app (no lifespan, no DB) ─────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)
_app.include_router(actions_route.router)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_action(
    action_id: str = "org.notify_churn",
    version: str = "1.0",
    namespace: Namespace = Namespace.ORG,
) -> ActionDefinition:
    return ActionDefinition(
        action_id=action_id,
        version=version,
        config=RegisterActionConfig(type="register", primitive_id="org.churn_flag"),
        trigger=TriggerConfig(
            fire_on=FireOn.ANY,
            condition_id="churn_risk",
            condition_version="1.0",
        ),
        namespace=namespace,
    )


def _make_store(actions: list[ActionDefinition], total: int | None = None) -> Any:
    """Return a DefinitionStore stub whose list_actions() returns the given list."""
    store = MagicMock()
    store.list_actions = AsyncMock(return_value=actions)
    # count_actions returns the DB total — defaults to len(actions) for existing tests.
    store.count_actions = AsyncMock(return_value=total if total is not None else len(actions))
    return store


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_list_actions_returns_actions_for_namespace():
    """GET /actions returns the list of ActionDefinitions for the given namespace."""
    actions = [
        _make_action("org.notify_churn", "1.0"),
        _make_action("org.send_webhook", "2.0"),
    ]
    stub_store = _make_store(actions)

    _app.dependency_overrides[get_definition_store] = lambda: stub_store

    try:
        with TestClient(_app) as client:
            resp = client.get("/actions", params={"namespace": "org"})
    finally:
        _app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["limit"] == 100
    assert body["offset"] == 0
    assert len(body["actions"]) == 2
    action_ids = {a["action_id"] for a in body["actions"]}
    assert action_ids == {"org.notify_churn", "org.send_webhook"}

    stub_store.list_actions.assert_awaited_once_with(
        namespace="org", limit=100, offset=0
    )


def test_list_actions_returns_empty_list_not_404():
    """GET /actions returns an empty list (not 404) when no actions exist."""
    stub_store = _make_store([])

    _app.dependency_overrides[get_definition_store] = lambda: stub_store

    try:
        with TestClient(_app) as client:
            resp = client.get("/actions", params={"namespace": "org"})
    finally:
        _app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["actions"] == []


def test_list_actions_excludes_deprecated():
    """
    GET /actions only returns non-deprecated actions.

    DefinitionStore.list_actions() enforces the deprecated=FALSE filter at the
    DB level. This test verifies the route passes through whatever the store
    returns — if the store returns an empty list, the route returns empty, not
    the deprecated actions.
    """
    # Store returns only non-deprecated (deprecated rows filtered in SQL)
    stub_store = _make_store([_make_action("org.active_action", "1.0")])

    _app.dependency_overrides[get_definition_store] = lambda: stub_store

    try:
        with TestClient(_app) as client:
            resp = client.get("/actions", params={"namespace": "org"})
    finally:
        _app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["actions"][0]["action_id"] == "org.active_action"


def test_list_actions_missing_namespace_returns_422():
    """GET /actions without a namespace query parameter returns HTTP 422."""
    stub_store = _make_store([])
    _app.dependency_overrides[get_definition_store] = lambda: stub_store

    try:
        with TestClient(_app) as client:
            resp = client.get("/actions")  # no namespace param
    finally:
        _app.dependency_overrides.clear()

    assert resp.status_code == 422


# ── FIX 6: total reflects DB count not page count ─────────────────────────────

def test_list_actions_total_reflects_db_count_not_page_count():
    """
    ActionListResponse.total must come from count_actions() (the DB total),
    NOT from len(actions) (the page count).

    This test gives the store 5 total actions in the DB but configures the
    route to return only 2 on the current page. Verifies that total=5, not
    total=2.
    """
    page_actions = [
        _make_action("org.action_1", "1.0"),
        _make_action("org.action_2", "1.0"),
    ]
    # DB has 5 total actions, but only 2 are on this page.
    stub_store = _make_store(page_actions, total=5)

    _app.dependency_overrides[get_definition_store] = lambda: stub_store

    try:
        with TestClient(_app) as client:
            resp = client.get("/actions", params={"namespace": "org", "limit": 2})
    finally:
        _app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5, (
        f"total must equal the DB count (5), not the page size (2). Got: {body['total']}"
    )
    assert len(body["actions"]) == 2

    # Verify count_actions was called with the correct namespace.
    stub_store.count_actions.assert_awaited_once_with(namespace="org")
