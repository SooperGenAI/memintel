"""
tests/integration/test_execute_static.py
──────────────────────────────────────────────────────────────────────────────
End-to-end HTTP tests for POST /execute/static.

Uses FastAPI's TestClient over ASGI — no real DB, no network.
The asyncpg pool is replaced with a mock whose fetchrow() returns
pre-built condition and concept body dicts (the same format stored
by DefinitionStore as JSONB in the definitions table).

Coverage:
  1. Threshold fires     — revenue=15000 > 10000 → value=True
  2. Threshold silent    — revenue=5000  < 10000 → value=False
  3. Equals fires        — account_tier="gold" matches → value="gold"
  4. Equals silent       — account_tier="bronze" → value=""
  5. Condition not found — 404 with error detail
  6. Concept not found   — 404 with error detail
  7. Response shape      — all required DecisionValue fields present
  8. Strategy type routed — z_score strategy returns boolean decision
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

# aioredis uses `distutils` removed in Python 3.12+; stub before any app import.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import execute as execute_route
from app.models.errors import MemintelError, memintel_error_handler
from app.persistence.db import get_db


# ── Shared definition bodies ────────────────────────────────────────────────
#
# These bodies mirror what DefinitionStore persists as JSONB. The concepts
# use missing_data_policy="zero" so that the primitive resolves to non-nullable
# float / categorical (required by z_score_op / passthrough type specs).

_CONCEPT_REVENUE = {
    "concept_id": "revenue_value",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "output_feature": "f_revenue",
    "primitives": {"revenue": {"type": "float", "missing_data_policy": "zero"}},
    "features":   {"f_revenue": {"op": "z_score_op", "inputs": {"input": "revenue"}}},
}

_CONCEPT_SEGMENT = {
    "concept_id": "account_segment",
    "version": "1.0",
    "namespace": "org",
    "output_type": "categorical",
    "labels": ["bronze", "silver", "gold"],
    "output_feature": "f_tier",
    "primitives": {"account_tier": {
        "type": "categorical",
        "labels": ["bronze", "silver", "gold"],
        "missing_data_policy": "forward_fill",
    }},
    "features": {"f_tier": {"op": "passthrough", "inputs": {"input": "account_tier"}}},
}

_CONDITION_THRESHOLD = {
    "condition_id": "high_revenue",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "revenue_value",
    "concept_version": "1.0",
    "strategy": {"type": "threshold", "params": {"direction": "above", "value": 10000}},
}

_CONDITION_EQUALS = {
    "condition_id": "is_gold_account",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "account_segment",
    "concept_version": "1.0",
    "strategy": {
        "type": "equals",
        "params": {"value": "gold", "labels": ["bronze", "silver", "gold"]},
    },
}


# ── Test app ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    """Skip all startup infrastructure (DB pool, Redis, config)."""
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)
_app.include_router(execute_route.router, prefix="/execute")


# ── Pool mock factory ─────────────────────────────────────────────────────────

def _make_pool(*rows) -> MagicMock:
    """
    Return a mock asyncpg pool whose fetchrow() yields each `row` in order.

    Each `row` is either a dict (returned as-is for row["key"] access) or None
    (simulates no matching DB record).
    """
    pool = MagicMock()
    pool.fetchrow = AsyncMock(side_effect=list(rows))
    return pool


def _row(body_dict: dict) -> dict:
    """Wrap a body dict in a fetchrow-compatible dict with a JSON-string body."""
    return {"body": json.dumps(body_dict)}


# ── Helper ────────────────────────────────────────────────────────────────────

def _post(client: TestClient, payload: dict) -> object:
    return client.post("/execute/static", json=payload)


# ── 1. Threshold fires ────────────────────────────────────────────────────────

class TestThresholdStrategy:
    def test_fires_when_revenue_above_threshold(self):
        pool = _make_pool(_row(_CONDITION_THRESHOLD), _row(_CONCEPT_REVENUE))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "acme_corp",
                    "data":              {"revenue": 15000},
                })
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["value"] is True
            assert body["condition_id"] == "high_revenue"
            assert body["condition_version"] == "1.0"
            assert body["entity"] == "acme_corp"
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_silent_when_revenue_below_threshold(self):
        pool = _make_pool(_row(_CONDITION_THRESHOLD), _row(_CONCEPT_REVENUE))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "small_co",
                    "data":              {"revenue": 5000},
                })
            assert r.status_code == 200, r.text
            assert r.json()["value"] is False
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_fires_exactly_at_boundary_is_false(self):
        """Threshold direction='above' is strictly greater than — boundary is False."""
        pool = _make_pool(_row(_CONDITION_THRESHOLD), _row(_CONCEPT_REVENUE))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "boundary_co",
                    "data":              {"revenue": 10000},
                })
            assert r.status_code == 200, r.text
            assert r.json()["value"] is False
        finally:
            _app.dependency_overrides.pop(get_db, None)


# ── 3-4. Equals strategy ──────────────────────────────────────────────────────

class TestEqualsStrategy:
    def test_fires_when_tier_matches(self):
        pool = _make_pool(_row(_CONDITION_EQUALS), _row(_CONCEPT_SEGMENT))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "is_gold_account",
                    "condition_version": "1.0",
                    "entity":            "acme_corp",
                    "data":              {"account_tier": "gold"},
                })
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["value"] == "gold"
            assert body["condition_id"] == "is_gold_account"
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_silent_when_tier_does_not_match(self):
        pool = _make_pool(_row(_CONDITION_EQUALS), _row(_CONCEPT_SEGMENT))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "is_gold_account",
                    "condition_version": "1.0",
                    "entity":            "small_co",
                    "data":              {"account_tier": "bronze"},
                })
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["value"] is None
            assert body["reason"] == "no_match"
        finally:
            _app.dependency_overrides.pop(get_db, None)


# ── 5. Condition not found ────────────────────────────────────────────────────

class TestNotFound:
    def test_condition_not_found_returns_404(self):
        pool = _make_pool(None)   # first fetchrow returns no row
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "no_such_condition",
                    "condition_version": "9.9",
                    "entity":            "any_entity",
                    "data":              {},
                })
            assert r.status_code == 404, r.text
            assert "not found" in r.json()["error"]["message"].lower()
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_concept_not_found_returns_404(self):
        # Condition found, concept row missing.
        pool = _make_pool(_row(_CONDITION_THRESHOLD), None)
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "any_entity",
                    "data":              {"revenue": 15000},
                })
            assert r.status_code == 404, r.text
            assert "not found" in r.json()["error"]["message"].lower()
        finally:
            _app.dependency_overrides.pop(get_db, None)


# ── 7. Response shape ─────────────────────────────────────────────────────────

class TestResponseShape:
    def test_all_decision_value_fields_present(self):
        pool = _make_pool(_row(_CONDITION_THRESHOLD), _row(_CONCEPT_REVENUE))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "acme_corp",
                    "data":              {"revenue": 15000},
                })
            assert r.status_code == 200, r.text
            body = r.json()
            for field in ("value", "decision_type", "condition_id",
                          "condition_version", "entity"):
                assert field in body, f"Missing field '{field}' in response: {body}"
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_decision_type_is_boolean_for_threshold(self):
        pool = _make_pool(_row(_CONDITION_THRESHOLD), _row(_CONCEPT_REVENUE))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "acme_corp",
                    "data":              {"revenue": 15000},
                })
            assert r.json()["decision_type"] == "boolean"
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_decision_type_is_categorical_for_equals(self):
        pool = _make_pool(_row(_CONDITION_EQUALS), _row(_CONCEPT_SEGMENT))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "is_gold_account",
                    "condition_version": "1.0",
                    "entity":            "acme_corp",
                    "data":              {"account_tier": "gold"},
                })
            assert r.json()["decision_type"] == "categorical"
        finally:
            _app.dependency_overrides.pop(get_db, None)

    def test_body_stored_as_json_string_is_parsed_correctly(self):
        """body stored as JSON string (common asyncpg JSONB return) is handled."""
        # _row() already wraps body as json.dumps() string — just confirm parsing.
        pool = _make_pool(_row(_CONDITION_THRESHOLD), _row(_CONCEPT_REVENUE))
        _app.dependency_overrides[get_db] = lambda: pool
        try:
            with TestClient(_app) as client:
                r = _post(client, {
                    "condition_id":      "high_revenue",
                    "condition_version": "1.0",
                    "entity":            "acme_corp",
                    "data":              {"revenue": 20000},
                })
            assert r.status_code == 200
            assert r.json()["value"] is True
        finally:
            _app.dependency_overrides.pop(get_db, None)
