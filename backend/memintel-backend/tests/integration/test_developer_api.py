"""
tests/integration/test_developer_api.py
────────────────────────────────────────────────────────────────────────────────
HTTP-level integration tests for the developer API routes.

Uses FastAPI's ASGI TestClient — no real DB, Redis, LLM, or network.
Each service-layer dependency is overridden with a scripted mock that returns
pre-built Pydantic model instances or raises typed MemintelError subclasses.

Route coverage
──────────────
Tasks:
  POST   /tasks                                → 200, Task
  POST   /tasks                                → 422, missing required fields
  GET    /tasks/{id}                           → 200, Task
  GET    /tasks/{id}                           → 404, unknown task_id
  PATCH  /tasks/{id}                           → 200, updated Task
  DELETE /tasks/{id}                           → 200, soft-deleted Task

Execution:
  POST   /evaluate/full                        → 200, FullPipelineResult
  POST   /evaluate/full  (dry_run=true)        → 200, actions status=would_trigger
  POST   /execute/batch                        → 200, one result per entity

Conditions:
  POST   /conditions/calibrate                 → 200, recommendation_available
  POST   /conditions/apply-calibration         → 200, new version created
  POST   /conditions/apply-calibration         → 400, second use of same token

Feedback:
  POST   /feedback/decision                    → 200, recorded
  POST   /feedback/decision                    → 409, duplicate submission

Registry:
  POST   /registry/definitions                 → 200, definition registered
  POST   /registry/definitions                 → 409, duplicate id+version
  GET    /registry/definitions/{id}/versions   → 404, unknown definition
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

# aioredis uses `distutils` which was removed in Python 3.12+.  Stub the
# package before any app module is imported so the import chain does not fail.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_elevated_key
from app.api.routes import conditions as conditions_route
from app.api.routes import execute as execute_route
from app.api.routes import feedback as feedback_route
from app.api.routes import registry as registry_route
from app.api.routes import tasks as tasks_route
from app.api.routes.conditions import get_calibration_service
from app.api.routes.execute import get_execute_service
from app.api.routes.feedback import get_feedback_service
from app.api.routes.registry import get_registry_service
from app.api.routes.tasks import get_task_authoring_service
from app.models.calibration import (
    ApplyCalibrationResult,
    CalibrationImpact,
    CalibrationResult,
    CalibrationStatus,
    FeedbackResponse,
    ImpactDirection,
    TaskPendingRebind,
)
from app.models.concept import DefinitionResponse
from app.models.condition import DecisionType
from app.models.errors import (
    ConflictError,
    ErrorType,
    MemintelError,
    NotFoundError,
    memintel_error_handler,
)
from app.models.result import (
    ActionTriggered,
    ActionTriggeredStatus,
    BatchExecuteItem,
    BatchExecuteResult,
    ConceptOutputType,
    ConceptResult,
    DecisionResult,
    FullPipelineResult,
)
from app.models.task import (
    DeliveryConfig,
    DeliveryType,
    Namespace,
    Task,
    TaskList,
    TaskStatus,
)
from app.persistence.stores import get_task_store


# ── Constants ──────────────────────────────────────────────────────────────────

_TASK_ID      = "task-dev-001"
_CONCEPT_ID   = "org.revenue_score"
_CONCEPT_V    = "1.0"
_COND_ID      = "cond.revenue_alert"
_COND_V1      = "1.0"
_COND_V2      = "1.1"
_ACTION_ID    = "action.email_alert"
_ACTION_V     = "1.0"
_ENTITY       = "customer_42"
_TOKEN        = "calib-tok-xyz789"
_DECISION_TS  = "2024-06-01T09:00:00Z"
_ELEVATED_KEY = "test-elevated-key"

_DELIVERY = DeliveryConfig(
    type=DeliveryType.WEBHOOK,
    endpoint="https://hooks.example.com/recv",
)

_CREATE_TASK_BODY = {
    "intent":       "alert when revenue score exceeds 0.8",
    "entity_scope": _ENTITY,
    "delivery":     {"type": "webhook", "endpoint": "https://hooks.example.com/recv"},
}

_EVAL_FULL_BODY = {
    "concept_id":        _CONCEPT_ID,
    "concept_version":   _CONCEPT_V,
    "condition_id":      _COND_ID,
    "condition_version": _COND_V1,
    "entity":            _ENTITY,
}

_FEEDBACK_BODY = {
    "condition_id":      _COND_ID,
    "condition_version": _COND_V1,
    "entity":            _ENTITY,
    "timestamp":         _DECISION_TS,
    "feedback":          "false_positive",
}

_REG_BODY = {
    "definition_id":   "org.test_rev_score",
    "version":         "1.0",
    "definition_type": "concept",
    "namespace":       "org",
    "body":            {"concept_id": "org.test_rev_score"},
}


# ── Shared model factories ─────────────────────────────────────────────────────

def _make_task(
    condition_version: str = _COND_V1,
    status: TaskStatus = TaskStatus.ACTIVE,
) -> Task:
    return Task(
        task_id=_TASK_ID,
        intent="alert when revenue score exceeds 0.8",
        concept_id=_CONCEPT_ID,
        concept_version=_CONCEPT_V,
        condition_id=_COND_ID,
        condition_version=condition_version,
        action_id=_ACTION_ID,
        action_version=_ACTION_V,
        entity_scope=_ENTITY,
        delivery=_DELIVERY,
        status=status,
    )


def _make_pipeline_result(*, dry_run: bool = False) -> FullPipelineResult:
    action_status = (
        ActionTriggeredStatus.WOULD_TRIGGER if dry_run
        else ActionTriggeredStatus.TRIGGERED
    )
    return FullPipelineResult(
        result=ConceptResult(
            value=0.85,
            type=ConceptOutputType.FLOAT,
            entity=_ENTITY,
            version=_CONCEPT_V,
            deterministic=True,
            timestamp=_DECISION_TS,
        ),
        decision=DecisionResult(
            value=True,
            type=DecisionType.BOOLEAN,
            entity=_ENTITY,
            condition_id=_COND_ID,
            condition_version=_COND_V1,
            timestamp=_DECISION_TS,
            actions_triggered=[
                ActionTriggered(
                    action_id=_ACTION_ID,
                    action_version=_ACTION_V,
                    status=action_status,
                )
            ],
        ),
        entity=_ENTITY,
        timestamp=_DECISION_TS,
        dry_run=dry_run,
    )


# ── Mock services ──────────────────────────────────────────────────────────────

class _MockTaskService:
    async def create_task(self, req: object) -> Task:
        return _make_task()

    async def update_task(self, task_id: str, body: object) -> Task:
        return _make_task(condition_version=_COND_V2)

    async def delete_task(self, task_id: str) -> Task:
        return _make_task(status=TaskStatus.DELETED)


class _MockTaskStoreHit:
    """Returns a Task for any known task_id (simulates a hit in the DB)."""

    async def get(self, task_id: str) -> Task | None:
        return _make_task()

    async def list(self, **kwargs: object) -> TaskList:
        return TaskList(
            items=[_make_task()],
            has_more=False,
            next_cursor=None,
            total_count=1,
        )


class _MockTaskStoreMiss:
    """Returns None for any task_id (simulates a DB miss)."""

    async def get(self, task_id: str) -> Task | None:
        return None


class _MockExecuteService:
    async def evaluate_full(self, req: object) -> FullPipelineResult:
        return _make_pipeline_result(dry_run=getattr(req, "dry_run", False))

    async def execute_batch(self, req: object) -> BatchExecuteResult:
        entities: list[str] = getattr(req, "entities", [])
        items = [
            BatchExecuteItem(
                entity=e,
                result=ConceptResult(
                    value=0.75,
                    type=ConceptOutputType.FLOAT,
                    entity=e,
                    version=_CONCEPT_V,
                    deterministic=False,
                ),
            )
            for e in entities
        ]
        return BatchExecuteResult(results=items, total=len(items), failed=0)


class _MockFeedbackServiceOk:
    async def submit(self, req: object) -> FeedbackResponse:
        return FeedbackResponse(status="recorded", feedback_id="fb-dev-001")


class _MockFeedbackServiceDuplicate:
    async def submit(self, req: object) -> FeedbackResponse:
        raise ConflictError("Feedback already submitted for this decision.")


class _MockCalibrationService:
    """
    First apply_calibration() call succeeds.
    Subsequent calls raise PARAMETER_ERROR (token already consumed).
    """

    def __init__(self) -> None:
        self._apply_count = 0

    async def calibrate(self, req: object) -> CalibrationResult:
        return CalibrationResult(
            status=CalibrationStatus.RECOMMENDATION_AVAILABLE,
            current_params={"value": 0.75},
            recommended_params={"value": 0.80},
            calibration_token=_TOKEN,
            impact=CalibrationImpact(
                delta_alerts=-0.5,
                direction=ImpactDirection.DECREASE,
            ),
        )

    async def apply_calibration(self, req: object) -> ApplyCalibrationResult:
        self._apply_count += 1
        if self._apply_count > 1:
            raise MemintelError(
                ErrorType.PARAMETER_ERROR,
                "Calibration token has already been used.",
            )
        return ApplyCalibrationResult(
            condition_id=_COND_ID,
            previous_version=_COND_V1,
            new_version=_COND_V2,
            params_applied={"value": 0.80},
            tasks_pending_rebind=[
                TaskPendingRebind(task_id=_TASK_ID, intent="alert when revenue score exceeds 0.8"),
            ],
        )


class _MockRegistryServiceOk:
    async def register(self, req: object) -> DefinitionResponse:
        return DefinitionResponse(
            definition_id=req.definition_id,
            version=req.version,
            definition_type=req.definition_type,
            namespace=Namespace.ORG,
        )

    async def get_versions(self, definition_id: str) -> list:
        raise NotFoundError(f"Definition '{definition_id}' not found.")


class _MockRegistryServiceConflict:
    async def register(self, req: object) -> DefinitionResponse:
        raise ConflictError("Definition already exists with a different body.")

    async def get_versions(self, definition_id: str) -> list:
        raise NotFoundError(f"Definition '{definition_id}' not found.")


# ── Test app ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    """No-op lifespan — skips all startup infrastructure (DB, Redis, config)."""
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)
_app.state.elevated_key = _ELEVATED_KEY          # required by require_elevated_key

_app.include_router(tasks_route.router)                                  # /tasks baked in
_app.include_router(execute_route.evaluate_router, prefix="/evaluate")   # /evaluate
_app.include_router(execute_route.router,          prefix="/execute")    # /execute
_app.include_router(feedback_route.router)                               # /feedback baked in
_app.include_router(conditions_route.router)                             # /conditions baked in
_app.include_router(registry_route.router,         prefix="/registry")   # /registry


# ── TASKS ──────────────────────────────────────────────────────────────────────

def test_create_task_200() -> None:
    """POST /tasks with valid required fields returns 200 with a Task."""
    _app.dependency_overrides[get_task_authoring_service] = lambda: _MockTaskService()
    try:
        with TestClient(_app) as client:
            r = client.post("/tasks", json=_CREATE_TASK_BODY)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["task_id"] == _TASK_ID
        assert body["condition_version"] == _COND_V1
        assert body["status"] == "active"
    finally:
        _app.dependency_overrides.clear()


def test_create_task_422_missing_required_fields() -> None:
    """
    POST /tasks with missing entity_scope and delivery returns 422.

    Pydantic request-body validation fires before any service dependency is
    resolved, so no service override is needed to observe the 422.
    """
    _app.dependency_overrides[get_task_authoring_service] = lambda: _MockTaskService()
    try:
        with TestClient(_app) as client:
            r = client.post("/tasks", json={"intent": "missing entity_scope and delivery"})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    finally:
        _app.dependency_overrides.clear()


def test_get_task_200() -> None:
    """GET /tasks/{id} for an existing task returns 200 with the Task."""
    _app.dependency_overrides[get_task_store] = lambda: _MockTaskStoreHit()
    try:
        with TestClient(_app) as client:
            r = client.get(f"/tasks/{_TASK_ID}")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        assert r.json()["task_id"] == _TASK_ID
    finally:
        _app.dependency_overrides.clear()


def test_get_task_404() -> None:
    """GET /tasks/{id} for an unknown task returns 404 with error.type=not_found."""
    _app.dependency_overrides[get_task_store] = lambda: _MockTaskStoreMiss()
    try:
        with TestClient(_app) as client:
            r = client.get("/tasks/does-not-exist")
        assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
        assert r.json()["error"]["type"] == "not_found"
    finally:
        _app.dependency_overrides.clear()


def test_update_task_200() -> None:
    """PATCH /tasks/{id} with condition_version rebind returns 200 with updated Task."""
    _app.dependency_overrides[get_task_authoring_service] = lambda: _MockTaskService()
    try:
        with TestClient(_app) as client:
            r = client.patch(
                f"/tasks/{_TASK_ID}",
                json={"condition_version": _COND_V2},
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        assert r.json()["condition_version"] == _COND_V2
    finally:
        _app.dependency_overrides.clear()


def test_delete_task_200() -> None:
    """DELETE /tasks/{id} returns 200 with status=deleted (soft delete)."""
    _app.dependency_overrides[get_task_authoring_service] = lambda: _MockTaskService()
    try:
        with TestClient(_app) as client:
            r = client.delete(f"/tasks/{_TASK_ID}")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        assert r.json()["status"] == "deleted"
    finally:
        _app.dependency_overrides.clear()


# ── EXECUTION ──────────────────────────────────────────────────────────────────

def test_evaluate_full_200() -> None:
    """POST /evaluate/full returns 200 with result and decision shapes."""
    _app.dependency_overrides[get_execute_service] = lambda: _MockExecuteService()
    try:
        with TestClient(_app) as client:
            r = client.post("/evaluate/full", json=_EVAL_FULL_BODY)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert "result" in body and "decision" in body
        assert body["result"]["value"] == pytest.approx(0.85)
        assert body["decision"]["value"] is True
    finally:
        _app.dependency_overrides.clear()


def test_evaluate_full_dry_run_would_trigger() -> None:
    """
    POST /evaluate/full with dry_run=true returns 200 and all actions
    carry status=would_trigger (actions were not actually fired).
    """
    _app.dependency_overrides[get_execute_service] = lambda: _MockExecuteService()
    try:
        with TestClient(_app) as client:
            r = client.post("/evaluate/full", json={**_EVAL_FULL_BODY, "dry_run": True})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body.get("dry_run") is True, "dry_run flag must be echoed in response"
        actions = body["decision"]["actions_triggered"]
        assert len(actions) >= 1, "Expected at least one action in dry_run result"
        for action in actions:
            assert action["status"] == "would_trigger", (
                f"All dry_run actions must be 'would_trigger', got: {action['status']}"
            )
    finally:
        _app.dependency_overrides.clear()


def test_execute_batch_200() -> None:
    """POST /execute/batch returns 200 with one result per entity in the request."""
    _app.dependency_overrides[get_execute_service] = lambda: _MockExecuteService()
    entities = ["ent_a", "ent_b", "ent_c"]
    try:
        with TestClient(_app) as client:
            r = client.post(
                "/execute/batch",
                json={"id": _CONCEPT_ID, "version": _CONCEPT_V, "entities": entities},
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["total"] == len(entities), (
            f"Expected total={len(entities)}, got {body['total']}"
        )
        assert body["failed"] == 0
        assert len(body["results"]) == len(entities)
    finally:
        _app.dependency_overrides.clear()


# ── CONDITIONS ─────────────────────────────────────────────────────────────────

def test_calibrate_condition_200() -> None:
    """POST /conditions/calibrate returns 200 with status=recommendation_available."""
    svc = _MockCalibrationService()
    _app.dependency_overrides[get_calibration_service] = lambda: svc
    try:
        with TestClient(_app) as client:
            r = client.post(
                "/conditions/calibrate",
                json={"condition_id": _COND_ID, "condition_version": _COND_V1},
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["status"] == "recommendation_available"
        assert body["calibration_token"] == _TOKEN
        assert "recommended_params" in body
    finally:
        _app.dependency_overrides.clear()


def test_apply_calibration_200() -> None:
    """POST /conditions/apply-calibration with a fresh token returns 200."""
    svc = _MockCalibrationService()
    _app.dependency_overrides[get_calibration_service] = lambda: svc
    try:
        with TestClient(_app) as client:
            r = client.post(
                "/conditions/apply-calibration",
                json={"calibration_token": _TOKEN},
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["new_version"] == _COND_V2
        assert body["condition_id"] == _COND_ID
        assert len(body["tasks_pending_rebind"]) == 1
    finally:
        _app.dependency_overrides.clear()


def test_apply_calibration_400_second_use() -> None:
    """
    POST /conditions/apply-calibration returns 400 on the second call with
    the same token — the token is single-use and is consumed on first apply.
    """
    svc = _MockCalibrationService()
    _app.dependency_overrides[get_calibration_service] = lambda: svc
    try:
        with TestClient(_app) as client:
            # First use must succeed.
            r1 = client.post(
                "/conditions/apply-calibration",
                json={"calibration_token": _TOKEN},
            )
            assert r1.status_code == 200, f"First apply failed: {r1.status_code} {r1.text}"

            # Second use of the same token must be rejected with HTTP 400.
            r2 = client.post(
                "/conditions/apply-calibration",
                json={"calibration_token": _TOKEN},
            )
        assert r2.status_code == 400, (
            f"Expected 400 on second token use, got {r2.status_code}: {r2.text}"
        )
        assert r2.json()["error"]["type"] == "parameter_error"
    finally:
        _app.dependency_overrides.clear()


# ── FEEDBACK ───────────────────────────────────────────────────────────────────

def test_submit_feedback_200() -> None:
    """POST /feedback/decision returns 200 with status=recorded and a feedback_id."""
    _app.dependency_overrides[get_feedback_service] = lambda: _MockFeedbackServiceOk()
    try:
        with TestClient(_app) as client:
            r = client.post("/feedback/decision", json=_FEEDBACK_BODY)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["status"] == "recorded"
        assert "feedback_id" in body
    finally:
        _app.dependency_overrides.clear()


def test_submit_feedback_409_duplicate() -> None:
    """POST /feedback/decision returns 409 on duplicate submission for the same decision."""
    _app.dependency_overrides[get_feedback_service] = lambda: _MockFeedbackServiceDuplicate()
    try:
        with TestClient(_app) as client:
            r = client.post("/feedback/decision", json=_FEEDBACK_BODY)
        assert r.status_code == 409, f"Expected 409, got {r.status_code}: {r.text}"
        assert r.json()["error"]["type"] == "conflict"
    finally:
        _app.dependency_overrides.clear()


# ── REGISTRY ───────────────────────────────────────────────────────────────────

def test_register_definition_200() -> None:
    """
    POST /registry/definitions with a valid payload and elevated key returns 200.

    require_elevated_key is overridden to a no-op so the test is not coupled
    to key configuration — security behaviour is exercised in test_security_scenarios.py.
    """
    _app.dependency_overrides[get_registry_service]  = lambda: _MockRegistryServiceOk()
    _app.dependency_overrides[require_elevated_key]  = lambda: None
    try:
        with TestClient(_app) as client:
            r = client.post(
                "/registry/definitions",
                json=_REG_BODY,
                headers={"X-Elevated-Key": _ELEVATED_KEY},
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["definition_id"] == _REG_BODY["definition_id"]
        assert body["version"] == _REG_BODY["version"]
    finally:
        _app.dependency_overrides.clear()


def test_register_definition_409_duplicate() -> None:
    """
    POST /registry/definitions returns 409 when (id, version) already exists
    with a different body — immutability violation.
    """
    _app.dependency_overrides[get_registry_service]  = lambda: _MockRegistryServiceConflict()
    _app.dependency_overrides[require_elevated_key]  = lambda: None
    try:
        with TestClient(_app) as client:
            r = client.post(
                "/registry/definitions",
                json=_REG_BODY,
                headers={"X-Elevated-Key": _ELEVATED_KEY},
            )
        assert r.status_code == 409, f"Expected 409, got {r.status_code}: {r.text}"
        assert r.json()["error"]["type"] == "conflict"
    finally:
        _app.dependency_overrides.clear()


def test_get_definition_versions_404() -> None:
    """
    GET /registry/definitions/{id}/versions returns 404 for an unknown definition.

    Maps to the GET /registry/{id} pattern from developer_api.yaml — the
    implementation exposes this via the /versions sub-path.
    """
    _app.dependency_overrides[get_registry_service] = lambda: _MockRegistryServiceOk()
    try:
        with TestClient(_app) as client:
            r = client.get("/registry/definitions/org.does-not-exist/versions")
        assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
        assert r.json()["error"]["type"] == "not_found"
    finally:
        _app.dependency_overrides.clear()
