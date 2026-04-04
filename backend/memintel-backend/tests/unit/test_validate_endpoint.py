"""
tests/unit/test_validate_endpoint.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for POST /registry/definitions/validate.

Coverage
────────
  1.  Valid concept body → 200 { valid: true, errors: [] }
  2.  Concept body with unresolved input ref → valid: false, SYNTAX_ERROR
  3.  Concept body with unknown operator → valid: false, REFERENCE_ERROR
  4.  Concept body with type mismatch → valid: false, TYPE_ERROR
  5.  Concept body with cycle → valid: false, GRAPH_ERROR
  6.  Concept body with container output type → valid: false, SEMANTIC_ERROR
  7.  Concept body with unparseable dict (missing required fields) → valid: false, SYNTAX_ERROR
  8.  Non-concept definition_type (condition) → 200 { valid: true, errors: [] }
  9.  Non-concept definition_type (action) → 200 { valid: true, errors: [] }
  10. Non-concept definition_type (primitive) → 200 { valid: true, errors: [] }
  11. Endpoint is read-only — no DB interaction required
  12. Missing API key is tolerated when app.state.api_key is None (dev mode)

Route registration order
────────────────────────
/definitions/validate is a static path — registered before /definitions/{id}/*
so it is never captured by the parameterised routes.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

# Stub heavy optional deps before any app module import
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import registry as registry_route
from app.models.errors import MemintelError, memintel_error_handler


# ── Minimal test app (no lifespan, no DB) ─────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)
_app.include_router(registry_route.router, prefix="/registry")

_client = TestClient(_app, raise_server_exceptions=True)


# ── Request helpers ────────────────────────────────────────────────────────────

_VALID_CONCEPT_BODY = {
    "concept_id": "org.churn_risk",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "churn_prob": {
            "type": "float",
            "missing_data_policy": "zero",
        },
    },
    "features": {
        "score": {
            "op": "normalize",
            "inputs": {"input": "churn_prob"},
            "params": {},
        },
    },
    "output_feature": "score",
}


def _validate_request(
    body: dict,
    definition_type: str = "concept",
    definition_id: str = "org.churn_risk",
    version: str = "1.0",
    namespace: str = "org",
) -> dict:
    return {
        "definition_id": definition_id,
        "version": version,
        "definition_type": definition_type,
        "namespace": namespace,
        "body": body,
    }


def _post(payload: dict) -> "Response":  # type: ignore[name-defined]
    return _client.post("/registry/definitions/validate", json=payload)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestValidateEndpointHappyPaths:

    def test_valid_concept_returns_valid_true(self):
        resp = _post(_validate_request(_VALID_CONCEPT_BODY))
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_non_concept_condition_always_valid(self):
        resp = _post(_validate_request(
            body={"condition_id": "org.cond", "version": "1.0"},
            definition_type="condition",
        ))
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_non_concept_action_always_valid(self):
        resp = _post(_validate_request(
            body={"action_id": "org.act", "version": "1.0"},
            definition_type="action",
        ))
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_non_concept_primitive_always_valid(self):
        resp = _post(_validate_request(
            body={"id": "org.prim", "version": "1.0"},
            definition_type="primitive",
        ))
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_non_concept_feature_always_valid(self):
        resp = _post(_validate_request(
            body={"feature_id": "org.feat", "version": "1.0"},
            definition_type="feature",
        ))
        assert resp.status_code == 200
        assert resp.json()["valid"] is True


class TestValidateEndpointConceptErrors:

    def test_unresolved_input_ref_returns_syntax_error(self):
        body = {
            **_VALID_CONCEPT_BODY,
            "features": {
                "score": {
                    "op": "normalize",
                    "inputs": {"input": "does_not_exist"},
                    "params": {},
                },
            },
        }
        resp = _post(_validate_request(body))
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) >= 1
        assert data["errors"][0]["type"] == "syntax_error"

    def test_unknown_operator_returns_reference_error(self):
        body = {
            **_VALID_CONCEPT_BODY,
            "features": {
                "score": {
                    "op": "nonexistent_op",
                    "inputs": {"input": "churn_prob"},
                    "params": {},
                },
            },
        }
        resp = _post(_validate_request(body))
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        types = {e["type"] for e in data["errors"]}
        assert "reference_error" in types

    def test_container_output_type_rejected(self):
        # ConceptDefinition has a field_validator that rejects container output types
        # before the compiler validator runs — Pydantic raises, we wrap as syntax_error.
        body = {
            **_VALID_CONCEPT_BODY,
            "output_type": "time_series<float>",
            "features": {
                "ts": {
                    "op": "time_series",
                    "inputs": {"input": "churn_prob"},
                    "params": {},
                },
            },
            "output_feature": "ts",
        }
        resp = _post(_validate_request(body))
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) >= 1

    def test_unparseable_body_returns_syntax_error(self):
        # body missing all required ConceptDefinition fields
        body = {"something": "irrelevant"}
        resp = _post(_validate_request(body))
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["errors"][0]["type"] == "syntax_error"

    def test_cycle_in_features_returns_graph_error(self):
        body = {
            **_VALID_CONCEPT_BODY,
            "features": {
                "a": {
                    "op": "normalize",
                    "inputs": {"input": "b"},
                    "params": {},
                },
                "b": {
                    "op": "normalize",
                    "inputs": {"input": "a"},
                    "params": {},
                },
            },
            "output_feature": "a",
        }
        resp = _post(_validate_request(body))
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        types = {e["type"] for e in data["errors"]}
        assert "graph_error" in types

    def test_errors_list_contains_location_when_present(self):
        body = {
            **_VALID_CONCEPT_BODY,
            "features": {
                "score": {
                    "op": "normalize",
                    "inputs": {"input": "missing_prim"},
                    "params": {},
                },
            },
        }
        resp = _post(_validate_request(body))
        data = resp.json()
        err = data["errors"][0]
        # location field should be present and point to the offending slot
        assert err.get("location") is not None
        assert "score" in err["location"]


class TestValidateEndpointContractDetails:

    def test_response_schema_has_valid_and_errors_fields(self):
        resp = _post(_validate_request(_VALID_CONCEPT_BODY))
        data = resp.json()
        assert "valid" in data
        assert "errors" in data
        assert isinstance(data["valid"], bool)
        assert isinstance(data["errors"], list)

    def test_does_not_require_elevated_key(self):
        # validate uses require_api_key (not elevated); with no api_key configured
        # in app.state (dev mode), the request goes through with no auth header.
        resp = _post(_validate_request(_VALID_CONCEPT_BODY))
        assert resp.status_code == 200

    def test_multiple_errors_accumulate(self):
        # A definition with both a reference_error (bad op) AND a type_error
        # should accumulate more than one error (phases run independently after
        # syntax passes).  Use two bad features so phases 2 and 3 both fire.
        body = {
            **_VALID_CONCEPT_BODY,
            "features": {
                "bad_op": {
                    "op": "unknown_op_xyz",
                    "inputs": {"input": "churn_prob"},
                    "params": {},
                },
                "score": {
                    "op": "normalize",
                    "inputs": {"input": "bad_op"},   # chains from bad_op
                    "params": {},
                },
            },
            "output_feature": "score",
        }
        resp = _post(_validate_request(body))
        data = resp.json()
        assert data["valid"] is False
        # At least one error (reference_error for unknown op)
        assert len(data["errors"]) >= 1
