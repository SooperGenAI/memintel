"""
tests/unit/test_fixes_round5.py
──────────────────────────────────────────────────────────────────────────────
Tests for the fifth round of bug fixes:

  FIX 1: NodeTrace.output_value and ConceptExplanation.output union ordering
          — bool | float | int | str — bool must come first so True is preserved
          as bool, not coerced to 1.0.

  FIX 2: ExecuteRequest and inline execute-route request models have max_length
          constraints on ID, version, and entity string fields.
          An oversized concept_id returns HTTP 422 before hitting any service.
"""
from __future__ import annotations

import inspect
from contextlib import asynccontextmanager

import pytest
import pydantic
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.errors import MemintelError, memintel_error_handler
from app.models.result import ConceptExplanation, ConceptOutputType, NodeTrace


# ── FIX 1: NodeTrace / ConceptExplanation bool union ordering ──────────────────

class TestNodeTraceOutputValueType:
    """NodeTrace.output_value=True must remain bool, not coerce to 1.0."""

    def test_bool_true_preserved(self):
        trace = NodeTrace(
            node_id="n1",
            op="AND",
            inputs={},
            params={},
            output_value=True,
            output_type="boolean",
        )
        assert trace.output_value is True
        assert type(trace.output_value) is bool

    def test_bool_false_preserved(self):
        trace = NodeTrace(
            node_id="n1",
            op="AND",
            inputs={},
            params={},
            output_value=False,
            output_type="boolean",
        )
        assert trace.output_value is False
        assert type(trace.output_value) is bool

    def test_bool_not_coerced_to_float(self):
        trace = NodeTrace(
            node_id="n1",
            op="AND",
            inputs={},
            params={},
            output_value=True,
            output_type="boolean",
        )
        # With wrong ordering (float | int | bool), True would become 1.0
        assert type(trace.output_value) is bool

    def test_float_preserved(self):
        trace = NodeTrace(
            node_id="n1",
            op="MUL",
            inputs={},
            params={},
            output_value=0.95,
            output_type="float",
        )
        assert trace.output_value == 0.95
        assert isinstance(trace.output_value, float)

    def test_str_preserved(self):
        trace = NodeTrace(
            node_id="n1",
            op="LABEL",
            inputs={},
            params={},
            output_value="high",
            output_type="categorical",
        )
        assert trace.output_value == "high"
        assert isinstance(trace.output_value, str)

    def test_model_validate_bool_true(self):
        data = {
            "node_id": "n1",
            "op": "AND",
            "inputs": {},
            "params": {},
            "output_value": True,
            "output_type": "boolean",
        }
        trace = NodeTrace.model_validate(data)
        assert trace.output_value is True
        assert type(trace.output_value) is bool


class TestConceptExplanationOutputType:
    """ConceptExplanation.output=True must remain bool, not coerce to 1.0."""

    def test_bool_true_preserved(self):
        expl = ConceptExplanation(output=True)
        assert expl.output is True
        assert type(expl.output) is bool

    def test_bool_false_preserved(self):
        expl = ConceptExplanation(output=False)
        assert expl.output is False
        assert type(expl.output) is bool

    def test_bool_not_coerced_to_float(self):
        expl = ConceptExplanation(output=True)
        assert type(expl.output) is bool

    def test_float_preserved(self):
        expl = ConceptExplanation(output=0.87)
        assert expl.output == 0.87
        assert isinstance(expl.output, float)

    def test_model_validate_bool(self):
        expl = ConceptExplanation.model_validate({"output": True})
        assert expl.output is True
        assert type(expl.output) is bool


# ── FIX 2: max_length on execute-route request models ─────────────────────────

class TestExecuteRequestMaxLength:
    """ExecuteRequest and inline route models reject oversized strings with 422."""

    # ── ExecuteRequest (from models/result.py) ─────────────────────────────────

    def test_execute_request_concept_id_too_long_raises_validation_error(self):
        from app.models.result import ExecuteRequest
        with pytest.raises(pydantic.ValidationError) as exc_info:
            ExecuteRequest(
                id="x" * 256,   # max_length=255
                version="1.0",
                entity="user:1",
            )
        errors = exc_info.value.errors()
        assert any("id" in str(e.get("loc", "")) for e in errors)

    def test_execute_request_version_too_long_raises_validation_error(self):
        from app.models.result import ExecuteRequest
        with pytest.raises(pydantic.ValidationError):
            ExecuteRequest(
                id="org.score",
                version="v" * 51,   # max_length=50
                entity="user:1",
            )

    def test_execute_request_entity_too_long_raises_validation_error(self):
        from app.models.result import ExecuteRequest
        with pytest.raises(pydantic.ValidationError):
            ExecuteRequest(
                id="org.score",
                version="1.0",
                entity="u" * 513,   # max_length=512
            )

    def test_execute_request_valid_passes(self):
        from app.models.result import ExecuteRequest
        req = ExecuteRequest(id="org.score", version="1.0", entity="user:1")
        assert req.id == "org.score"

    # ── HTTP-level 422 via route handler ────────────────────────────────────────

    @pytest.fixture(autouse=True)
    def _route_app(self):
        from app.api.routes import execute as execute_route
        from app.api.routes.execute import get_execute_service
        from app.persistence.db import get_db

        @asynccontextmanager
        async def _null_lifespan(app: FastAPI):
            yield

        async def _null_db():
            return None

        async def _null_service():
            return None

        app = FastAPI(lifespan=_null_lifespan)
        app.add_exception_handler(MemintelError, memintel_error_handler)
        app.include_router(execute_route.router, prefix="/execute")
        app.include_router(execute_route.evaluate_router, prefix="/evaluate")
        app.dependency_overrides[get_db] = _null_db
        app.dependency_overrides[get_execute_service] = _null_service
        self._app = app
        yield
        app.dependency_overrides.clear()

    def test_evaluate_full_concept_id_too_long_returns_422(self):
        with TestClient(self._app) as client:
            resp = client.post(
                "/evaluate/full",
                json={
                    "concept_id": "x" * 256,
                    "concept_version": "1.0",
                    "condition_id": "high_churn",
                    "condition_version": "1.0",
                    "entity": "user:1",
                },
            )
        assert resp.status_code == 422

    def test_evaluate_full_entity_too_long_returns_422(self):
        with TestClient(self._app) as client:
            resp = client.post(
                "/evaluate/full",
                json={
                    "concept_id": "org.score",
                    "concept_version": "1.0",
                    "condition_id": "high_churn",
                    "condition_version": "1.0",
                    "entity": "u" * 513,
                },
            )
        assert resp.status_code == 422

    def test_evaluate_condition_condition_id_too_long_returns_422(self):
        with TestClient(self._app) as client:
            resp = client.post(
                "/evaluate/condition",
                json={
                    "condition_id": "c" * 256,
                    "condition_version": "1.0",
                    "entity": "user:1",
                },
            )
        assert resp.status_code == 422

    # ── EvaluateFullRequest model directly ─────────────────────────────────────

    def test_evaluate_full_request_model_rejects_long_concept_id(self):
        from app.api.routes.execute import EvaluateFullRequest
        with pytest.raises(pydantic.ValidationError):
            EvaluateFullRequest(
                concept_id="x" * 256,
                concept_version="1.0",
                condition_id="high_churn",
                condition_version="1.0",
                entity="user:1",
            )

    def test_evaluate_full_request_model_rejects_long_version(self):
        from app.api.routes.execute import EvaluateFullRequest
        with pytest.raises(pydantic.ValidationError):
            EvaluateFullRequest(
                concept_id="org.score",
                concept_version="v" * 51,
                condition_id="high_churn",
                condition_version="1.0",
                entity="user:1",
            )

    def test_evaluate_full_request_model_valid(self):
        from app.api.routes.execute import EvaluateFullRequest
        req = EvaluateFullRequest(
            concept_id="org.score",
            concept_version="1.0",
            condition_id="high_churn",
            condition_version="1.0",
            entity="user:1",
        )
        assert req.concept_id == "org.score"
