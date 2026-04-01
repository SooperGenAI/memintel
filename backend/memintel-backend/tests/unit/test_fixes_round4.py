"""
tests/unit/test_fixes_round4.py
──────────────────────────────────────────────────────────────────────────────
Tests for the fourth round of bug fixes:

  FIX 1: DecisionRecord.concept_value union ordering — bool | float | int | str | None
          ensures value=True is preserved as bool, not coerced to 1.0.

  FIX 2: RegisterDefinitionRequest rejects invalid definition_type with a Pydantic
          422-style ValidationError before the request reaches the DB CHECK constraint.

  FIX 3: _async_with_retry() uses asyncio.sleep(), not time.sleep(), so the event
          loop is not blocked during connector backoff.

  FIX 5: GET /tasks with an invalid status value returns HTTP 422 (Pydantic enum
          validation), not a silent empty list or DB error.
"""
from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pydantic
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.decision import DecisionRecord
from app.models.errors import MemintelError, memintel_error_handler


# ── FIX 1: DecisionRecord bool union ordering ──────────────────────────────────

class TestDecisionRecordConceptValueType:
    """concept_value=True must remain bool, not be coerced to 1.0 (float)."""

    def _make_record(self, concept_value: Any) -> DecisionRecord:
        return DecisionRecord(
            concept_id="org.churn_score",
            concept_version="1.0",
            condition_id="high_churn",
            condition_version="1.0",
            entity_id="user:1",
            fired=True,
            concept_value=concept_value,
        )

    def test_bool_true_preserved_as_bool(self):
        rec = self._make_record(True)
        assert rec.concept_value is True
        assert type(rec.concept_value) is bool

    def test_bool_false_preserved_as_bool(self):
        rec = self._make_record(False)
        assert rec.concept_value is False
        assert type(rec.concept_value) is bool

    def test_bool_true_not_coerced_to_float(self):
        rec = self._make_record(True)
        # With wrong ordering (float | int | bool), True would become 1.0
        assert rec.concept_value != 1.0 or type(rec.concept_value) is bool
        assert type(rec.concept_value) is bool

    def test_float_value_preserved(self):
        rec = self._make_record(0.75)
        assert rec.concept_value == 0.75
        assert isinstance(rec.concept_value, float)

    def test_int_value_preserved(self):
        rec = self._make_record(42)
        assert rec.concept_value == 42
        assert not isinstance(rec.concept_value, bool)

    def test_str_value_preserved(self):
        rec = self._make_record("high")
        assert rec.concept_value == "high"
        assert isinstance(rec.concept_value, str)

    def test_none_value_preserved(self):
        rec = self._make_record(None)
        assert rec.concept_value is None

    def test_model_validate_bool(self):
        data = {
            "concept_id": "org.churn_score",
            "concept_version": "1.0",
            "condition_id": "high_churn",
            "condition_version": "1.0",
            "entity_id": "user:1",
            "fired": True,
            "concept_value": True,
        }
        rec = DecisionRecord.model_validate(data)
        assert rec.concept_value is True
        assert type(rec.concept_value) is bool


# ── FIX 2: RegisterDefinitionRequest validation ────────────────────────────────

class TestRegisterDefinitionRequestValidation:
    """definition_type must be one of the Literal values; invalid → ValidationError."""

    def _make_req(self, definition_type: str, namespace: str = "org") -> Any:
        from app.api.routes.registry import RegisterDefinitionRequest
        return RegisterDefinitionRequest(
            definition_id="org.score",
            version="1.0",
            definition_type=definition_type,
            namespace=namespace,
            body={"concept_id": "org.score"},
        )

    def test_valid_concept_accepted(self):
        req = self._make_req("concept")
        assert req.definition_type == "concept"

    def test_valid_condition_accepted(self):
        req = self._make_req("condition")
        assert req.definition_type == "condition"

    def test_valid_action_accepted(self):
        req = self._make_req("action")
        assert req.definition_type == "action"

    def test_valid_primitive_accepted(self):
        req = self._make_req("primitive")
        assert req.definition_type == "primitive"

    def test_valid_feature_accepted(self):
        req = self._make_req("feature")
        assert req.definition_type == "feature"

    def test_invalid_definition_type_raises_validation_error(self):
        with pytest.raises(pydantic.ValidationError) as exc_info:
            self._make_req("trigger")
        errors = exc_info.value.errors()
        assert any("definition_type" in str(e.get("loc", "")) for e in errors)

    def test_unknown_type_raises_validation_error(self):
        with pytest.raises(pydantic.ValidationError):
            self._make_req("foobar")

    def test_empty_type_raises_validation_error(self):
        with pytest.raises(pydantic.ValidationError):
            self._make_req("")

    def test_namespace_whitespace_stripped(self):
        req = self._make_req("concept", namespace="  org  ")
        assert req.namespace == "org"

    def test_namespace_lowercased(self):
        req = self._make_req("concept", namespace="ORG")
        assert req.namespace == "org"

    def test_namespace_stripped_and_lowercased(self):
        req = self._make_req("concept", namespace="  PERSONAL  ")
        assert req.namespace == "personal"


# ── FIX 3: _async_with_retry uses asyncio.sleep ────────────────────────────────

class TestAsyncWithRetry:
    """_async_with_retry() must use asyncio.sleep, not time.sleep."""

    def test_async_with_retry_is_coroutine_function(self):
        from app.runtime.data_resolver import _async_with_retry
        assert inspect.iscoroutinefunction(_async_with_retry)

    @pytest.mark.asyncio
    async def test_async_with_retry_succeeds_immediately(self):
        from app.runtime.data_resolver import _async_with_retry
        result = await _async_with_retry(lambda: 42, max_retries=0, backoff_base=0.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_with_retry_uses_asyncio_sleep_not_time_sleep(self):
        """Verify asyncio.sleep is called (not time.sleep) on transient failure."""
        from app.runtime.data_resolver import (
            _async_with_retry,
            TransientConnectorError,
        )

        call_count = 0

        def _failing_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TransientConnectorError("transient")
            return "ok"

        sleep_calls: list[float] = []

        async def _fake_asyncio_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("app.runtime.data_resolver.asyncio.sleep", side_effect=_fake_asyncio_sleep):
            with patch("app.runtime.data_resolver.time.sleep") as mock_time_sleep:
                result = await _async_with_retry(
                    _failing_then_ok, max_retries=2, backoff_base=0.1
                )

        assert result == "ok"
        # asyncio.sleep was called for the backoff
        assert len(sleep_calls) >= 1
        # time.sleep was NOT called
        mock_time_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_with_retry_reraises_after_all_retries_exhausted(self):
        from app.runtime.data_resolver import (
            _async_with_retry,
            TransientConnectorError,
        )

        def _always_fail():
            raise TransientConnectorError("always fails")

        with pytest.raises(TransientConnectorError):
            await _async_with_retry(_always_fail, max_retries=1, backoff_base=0.0)

    @pytest.mark.asyncio
    async def test_async_with_retry_does_not_retry_auth_error(self):
        from app.runtime.data_resolver import (
            _async_with_retry,
            AuthConnectorError,
        )
        call_count = 0

        def _always_auth_fail():
            nonlocal call_count
            call_count += 1
            raise AuthConnectorError("auth failure")

        with pytest.raises(AuthConnectorError):
            await _async_with_retry(_always_auth_fail, max_retries=3, backoff_base=0.0)

        # Should have been called exactly once (no retries for auth failures)
        assert call_count == 1


# ── FIX 5: GET /tasks invalid status returns 422 ──────────────────────────────

class TestListTasksStatusValidation:
    """GET /tasks with an invalid status value must return HTTP 422."""

    @pytest.fixture(autouse=True)
    def _app(self):
        from app.api.routes import tasks as tasks_route
        from app.api.routes.tasks import get_task_store

        @asynccontextmanager
        async def _null_lifespan(app: FastAPI):
            yield

        app = FastAPI(lifespan=_null_lifespan)
        app.add_exception_handler(MemintelError, memintel_error_handler)
        app.include_router(tasks_route.router)

        stub_store = MagicMock()
        stub_store.list = AsyncMock(return_value=MagicMock(items=[], has_more=False, next_cursor=None))

        app.dependency_overrides[get_task_store] = lambda: stub_store
        self._stub_store = stub_store
        self._client_app = app
        yield
        app.dependency_overrides.clear()

    def test_invalid_status_returns_422(self):
        with TestClient(self._client_app) as client:
            resp = client.get("/tasks", params={"status": "flying"})
        assert resp.status_code == 422

    def test_another_invalid_status_returns_422(self):
        with TestClient(self._client_app) as client:
            resp = client.get("/tasks", params={"status": "running"})
        assert resp.status_code == 422

    def test_valid_status_active_accepted(self):
        from app.models.task import TaskList
        self._stub_store.list = AsyncMock(
            return_value=TaskList(items=[], has_more=False, next_cursor=None, total_count=0)
        )
        with TestClient(self._client_app) as client:
            resp = client.get("/tasks", params={"status": "active"})
        assert resp.status_code == 200

    def test_valid_status_deleted_accepted(self):
        from app.models.task import TaskList
        self._stub_store.list = AsyncMock(
            return_value=TaskList(items=[], has_more=False, next_cursor=None, total_count=0)
        )
        with TestClient(self._client_app) as client:
            resp = client.get("/tasks", params={"status": "deleted"})
        assert resp.status_code == 200

    def test_no_status_param_accepted(self):
        from app.models.task import TaskList
        self._stub_store.list = AsyncMock(
            return_value=TaskList(items=[], has_more=False, next_cursor=None, total_count=0)
        )
        with TestClient(self._client_app) as client:
            resp = client.get("/tasks")
        assert resp.status_code == 200
