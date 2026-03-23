"""
tests/integration/test_http_happy_path.py
──────────────────────────────────────────────────────────────────────────────
HTTP happy path — 7-step end-to-end verification using FastAPI's TestClient
over ASGI (no real network, no DB, no Redis, no LLM).

Each service-layer dependency is overridden with a scripted mock that returns
pre-built Pydantic response models.  The test verifies:

  Step 1  POST /tasks                         → 200, Task; capture task_id
  Step 2  POST /evaluate/full                 → 200, FullPipelineResult shape
  Step 3  POST /feedback/decision             → 200, FeedbackResponse recorded
  Step 4  POST /conditions/calibrate          → 200, CalibrationResult + token
  Step 5  POST /conditions/apply-calibration  → 200, ApplyCalibrationResult v1.1
  Step 6  PATCH /tasks/{id}                   → 200, Task condition_version=1.1
  Step 7  POST /evaluate/full (again)         → 200, decision.condition_version=1.1

State flows between steps:
  task_id           — step 1 → step 6 URL
  calibration_token — step 4 → step 5 body
  new_version       — step 5 → step 6 body and step 7 request

Note on route paths:
  /conditions/apply-calibration matches the route definition in
  app/api/routes/conditions.py.  The developer_api.yaml spec uses the path
  /conditions/calibrate/apply — this divergence is noted below.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

# aioredis uses `distutils` which was removed in Python 3.12+.  Stub the
# package before any app module is imported so the import chain does not fail.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import conditions as conditions_route
from app.api.routes import execute as execute_route
from app.api.routes import feedback as feedback_route
from app.api.routes import tasks as tasks_route
from app.api.routes.conditions import get_calibration_service
from app.api.routes.execute import get_execute_service
from app.api.routes.feedback import get_feedback_service
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
from app.models.condition import DecisionType
from app.models.errors import MemintelError, memintel_error_handler
from app.models.result import (
    ConceptOutputType,
    ConceptResult,
    DecisionResult,
    FullPipelineResult,
)
from app.models.task import DeliveryConfig, DeliveryType, Task, TaskStatus


# ── Shared test constants ──────────────────────────────────────────────────────

_TASK_ID       = "task-001"
_CONDITION_ID  = "cond.churn_alert"
_COND_V1       = "1.0"
_COND_V2       = "1.1"
_CONCEPT_ID    = "org.churn_risk_score"
_CONCEPT_V     = "1.0"
_ACTION_ID     = "action.slack_alert"
_ACTION_V      = "1.0"
_ENTITY        = "org_123"
_TOKEN         = "calib-token-abc123"
_INTENT        = "alert when churn risk score exceeds 0.7"
_DECISION_TS   = "2024-01-15T12:00:00Z"

_DELIVERY = DeliveryConfig(
    type=DeliveryType.WEBHOOK,
    endpoint="https://example.com/hook",
)


# ── Helpers to build model instances ──────────────────────────────────────────

def _make_task(condition_version: str) -> Task:
    return Task(
        task_id=_TASK_ID,
        intent=_INTENT,
        concept_id=_CONCEPT_ID,
        concept_version=_CONCEPT_V,
        condition_id=_CONDITION_ID,
        condition_version=condition_version,
        action_id=_ACTION_ID,
        action_version=_ACTION_V,
        entity_scope=_ENTITY,
        delivery=_DELIVERY,
        status=TaskStatus.ACTIVE,
    )


def _make_pipeline_result(condition_version: str) -> FullPipelineResult:
    return FullPipelineResult(
        result=ConceptResult(
            value=0.82,
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
            condition_id=_CONDITION_ID,
            condition_version=condition_version,
            timestamp=_DECISION_TS,
            actions_triggered=[],
        ),
        entity=_ENTITY,
        timestamp=_DECISION_TS,
    )


# ── Scripted mock service classes ──────────────────────────────────────────────

class _MockTaskService:
    """
    Records every call to create_task() and update_task() and returns
    pre-built Task instances pinned to the expected condition versions.
    """

    def __init__(self) -> None:
        self.create_calls: list[object] = []
        self.update_calls: list[tuple[str, object]] = []

    async def create_task(self, req: object) -> Task:
        self.create_calls.append(req)
        return _make_task(_COND_V1)

    async def update_task(self, task_id: str, body: object) -> Task:
        self.update_calls.append((task_id, body))
        return _make_task(_COND_V2)


class _MockExecuteService:
    """
    Records every evaluate_full() call and echoes the requested
    condition_version back inside the DecisionResult so that step 7 can
    assert the correct version was used.
    """

    def __init__(self) -> None:
        self.evaluate_full_calls: list[object] = []

    async def evaluate_full(self, req: object) -> FullPipelineResult:
        self.evaluate_full_calls.append(req)
        cv = getattr(req, "condition_version", _COND_V1)
        return _make_pipeline_result(cv)


class _MockFeedbackService:
    """Returns a fixed FeedbackResponse — no DB access required."""

    async def submit(self, req: object) -> FeedbackResponse:
        return FeedbackResponse(status="recorded", feedback_id="fb-001")


class _MockCalibrationService:
    """
    calibrate() returns a recommendation with a known token.
    apply_calibration() returns a new version "1.1" and lists the task as
    pending rebind — matching the state established in step 1.
    """

    def __init__(self) -> None:
        self.calibrate_calls: list[object] = []
        self.apply_calls: list[object] = []

    async def calibrate(self, req: object) -> CalibrationResult:
        self.calibrate_calls.append(req)
        return CalibrationResult(
            status=CalibrationStatus.RECOMMENDATION_AVAILABLE,
            current_params={"value": 0.70},
            recommended_params={"value": 0.77},
            calibration_token=_TOKEN,
            impact=CalibrationImpact(
                delta_alerts=-1.0,
                direction=ImpactDirection.DECREASE,
            ),
        )

    async def apply_calibration(self, req: object) -> ApplyCalibrationResult:
        self.apply_calls.append(req)
        return ApplyCalibrationResult(
            condition_id=_CONDITION_ID,
            previous_version=_COND_V1,
            new_version=_COND_V2,
            params_applied={"value": 0.77},
            tasks_pending_rebind=[
                TaskPendingRebind(task_id=_TASK_ID, intent=_INTENT),
            ],
        )


# ── Test app (null lifespan — no DB / Redis / config file needed) ──────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    """No-op lifespan that skips all startup infrastructure requirements."""
    yield


_app = FastAPI(lifespan=_null_lifespan)
_app.add_exception_handler(MemintelError, memintel_error_handler)

_app.include_router(tasks_route.router)                                         # prefix="/tasks" baked in
_app.include_router(execute_route.evaluate_router,  prefix="/evaluate")     # no prefix on router
_app.include_router(execute_route.router,           prefix="/execute")      # no prefix on router
_app.include_router(feedback_route.router)                                      # prefix="/feedback" baked in
_app.include_router(conditions_route.router)                                    # prefix="/conditions" baked in


# ── Test ───────────────────────────────────────────────────────────────────────

def test_happy_path_seven_steps() -> None:
    """
    Verify the complete 7-step happy path via HTTP.

    Each step asserts:
      - HTTP 200 status code
      - Expected response shape (presence and values of key fields)

    Inter-step state:
      task_id           — step 1 → step 6 URL
      calibration_token — step 4 → step 5 body
      new_version       — step 5 → step 6 body and step 7 request

    Post-loop invariants verify that each mock was called exactly the
    expected number of times.
    """
    # ── Fresh mock instances — isolates this test from any future siblings ─────
    task_svc       = _MockTaskService()
    execute_svc    = _MockExecuteService()
    feedback_svc   = _MockFeedbackService()
    calibration_svc = _MockCalibrationService()

    _app.dependency_overrides[get_task_authoring_service] = lambda: task_svc
    _app.dependency_overrides[get_execute_service]        = lambda: execute_svc
    _app.dependency_overrides[get_feedback_service]       = lambda: feedback_svc
    _app.dependency_overrides[get_calibration_service]    = lambda: calibration_svc

    try:
        with TestClient(_app) as client:

            # ── Step 1: Create task ──────────────────────────────────────────
            r1 = client.post(
                "/tasks",
                json={
                    "intent":       _INTENT,
                    "entity_scope": _ENTITY,
                    "delivery":     {"type": "webhook", "endpoint": "https://example.com/hook"},
                },
            )
            assert r1.status_code == 200, f"Step 1 — POST /tasks: {r1.status_code} {r1.text}"
            t1 = r1.json()
            assert t1["task_id"] == _TASK_ID, f"Step 1 — unexpected task_id: {t1}"
            assert t1["condition_version"] == _COND_V1, \
                f"Step 1 — expected condition_version={_COND_V1}, got {t1['condition_version']}"
            task_id = t1["task_id"]

            # ── Step 2: Execute full pipeline (condition v1.0) ───────────────
            r2 = client.post(
                "/evaluate/full",
                json={
                    "concept_id":        _CONCEPT_ID,
                    "concept_version":   _CONCEPT_V,
                    "condition_id":      _CONDITION_ID,
                    "condition_version": _COND_V1,
                    "entity":            _ENTITY,
                },
            )
            assert r2.status_code == 200, \
                f"Step 2 — POST /evaluate/full: {r2.status_code} {r2.text}"
            fp2 = r2.json()
            assert "result"   in fp2, f"Step 2 — 'result' key missing from response: {fp2}"
            assert "decision" in fp2, f"Step 2 — 'decision' key missing from response: {fp2}"
            assert abs(fp2["result"]["value"] - 0.82) < 1e-6, \
                f"Step 2 — unexpected result.value: {fp2['result']['value']}"
            assert fp2["decision"]["value"] is True, \
                f"Step 2 — unexpected decision.value: {fp2['decision']['value']}"
            assert fp2["decision"]["condition_version"] == _COND_V1, \
                f"Step 2 — wrong condition_version in decision: {fp2['decision']['condition_version']}"

            # ── Step 3: Submit false_positive feedback ───────────────────────
            r3 = client.post(
                "/feedback/decision",
                json={
                    "condition_id":      _CONDITION_ID,
                    "condition_version": _COND_V1,
                    "entity":            _ENTITY,
                    "timestamp":         _DECISION_TS,
                    "feedback":          "false_positive",
                },
            )
            assert r3.status_code == 200, \
                f"Step 3 — POST /feedback/decision: {r3.status_code} {r3.text}"
            fb3 = r3.json()
            assert fb3["status"] == "recorded", \
                f"Step 3 — unexpected status: {fb3['status']}"
            assert "feedback_id" in fb3, f"Step 3 — feedback_id missing: {fb3}"

            # ── Step 4: Request calibration recommendation ───────────────────
            r4 = client.post(
                "/conditions/calibrate",
                json={
                    "condition_id":       _CONDITION_ID,
                    "condition_version":  _COND_V1,
                    "feedback_direction": "tighten",
                },
            )
            assert r4.status_code == 200, \
                f"Step 4 — POST /conditions/calibrate: {r4.status_code} {r4.text}"
            cal4 = r4.json()
            assert cal4["status"] == "recommendation_available", \
                f"Step 4 — expected recommendation_available, got {cal4['status']}"
            assert cal4.get("calibration_token") == _TOKEN, \
                f"Step 4 — wrong calibration_token: {cal4.get('calibration_token')}"
            assert "recommended_params" in cal4, f"Step 4 — recommended_params missing: {cal4}"
            calibration_token = cal4["calibration_token"]

            # ── Step 5: Apply calibration — creates condition v1.1 ───────────
            #
            # Route path: POST /conditions/apply-calibration
            # (developer_api.yaml uses /conditions/calibrate/apply — the
            # implementation diverges; see conditions.py router definition.)
            r5 = client.post(
                "/conditions/apply-calibration",
                json={"calibration_token": calibration_token},
            )
            assert r5.status_code == 200, \
                f"Step 5 — POST /conditions/apply-calibration: {r5.status_code} {r5.text}"
            app5 = r5.json()
            assert app5["new_version"] == _COND_V2, \
                f"Step 5 — expected new_version={_COND_V2}, got {app5['new_version']}"
            assert app5["condition_id"] == _CONDITION_ID, \
                f"Step 5 — wrong condition_id: {app5['condition_id']}"
            assert len(app5["tasks_pending_rebind"]) == 1, \
                f"Step 5 — expected 1 pending task, got {len(app5['tasks_pending_rebind'])}"
            assert app5["tasks_pending_rebind"][0]["task_id"] == _TASK_ID, \
                f"Step 5 — wrong task_id in pending rebind: {app5['tasks_pending_rebind'][0]}"
            new_version = app5["new_version"]

            # ── Step 6: Rebind task to new condition version ─────────────────
            r6 = client.patch(
                f"/tasks/{task_id}",
                json={"condition_version": new_version},
            )
            assert r6.status_code == 200, \
                f"Step 6 — PATCH /tasks/{task_id}: {r6.status_code} {r6.text}"
            t6 = r6.json()
            assert t6["condition_version"] == _COND_V2, \
                f"Step 6 — expected condition_version={_COND_V2}, got {t6['condition_version']}"
            assert t6["task_id"] == _TASK_ID, \
                f"Step 6 — task_id changed unexpectedly: {t6['task_id']}"

            # ── Step 7: Execute full pipeline (condition v1.1) ───────────────
            r7 = client.post(
                "/evaluate/full",
                json={
                    "concept_id":        _CONCEPT_ID,
                    "concept_version":   _CONCEPT_V,
                    "condition_id":      _CONDITION_ID,
                    "condition_version": _COND_V2,
                    "entity":            _ENTITY,
                },
            )
            assert r7.status_code == 200, \
                f"Step 7 — POST /evaluate/full (v2): {r7.status_code} {r7.text}"
            fp7 = r7.json()
            assert fp7["decision"]["condition_version"] == _COND_V2, (
                f"Step 7 — evaluate/full was called with the old condition_version "
                f"(expected {_COND_V2}, got "
                f"'{fp7['decision']['condition_version']}')"
            )

        # ── Post-loop invariants ───────────────────────────────────────────────
        # create_task called exactly once (step 1).
        assert len(task_svc.create_calls) == 1, \
            f"Expected 1 create_task call, got {len(task_svc.create_calls)}"

        # update_task called exactly once (step 6) with the correct task_id.
        assert len(task_svc.update_calls) == 1, \
            f"Expected 1 update_task call, got {len(task_svc.update_calls)}"
        assert task_svc.update_calls[0][0] == _TASK_ID, \
            f"update_task was called with wrong task_id: {task_svc.update_calls[0][0]}"

        # evaluate_full called exactly twice: once at v1 (step 2), once at v2 (step 7).
        assert len(execute_svc.evaluate_full_calls) == 2, \
            f"Expected 2 evaluate_full calls, got {len(execute_svc.evaluate_full_calls)}"

        # calibrate called once (step 4); apply_calibration called once (step 5).
        assert len(calibration_svc.calibrate_calls) == 1, \
            f"Expected 1 calibrate call, got {len(calibration_svc.calibrate_calls)}"
        assert len(calibration_svc.apply_calls) == 1, \
            f"Expected 1 apply_calibration call, got {len(calibration_svc.apply_calls)}"

    finally:
        # Restore dependency_overrides so the app is clean for any future tests.
        _app.dependency_overrides.clear()
