"""
tests/integration/run_http_happy_path.py
──────────────────────────────────────────────────────────────────────────────
Boots the Memintel test app on an OS-assigned local TCP port using uvicorn,
then fires real HTTP calls (actual TCP socket) for each of the 7 happy-path
steps defined in developer_api.yaml.

No database, Redis, or LLM required.  Service-layer dependencies are overridden
with the same scripted mock objects as in test_http_happy_path.py.

Run:
    python tests/integration/run_http_happy_path.py

Exit code 0 -> all steps returned expected status codes and response shapes.
Exit code 1 -> one or more steps failed (details printed to stdout).
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

# ── Stub aioredis before any app import ──────────────────────────────────────
sys.modules.setdefault("aioredis", MagicMock())

import httpx
import uvicorn
from fastapi import FastAPI

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
from app.models.result import ConceptOutputType, ConceptResult, DecisionResult, FullPipelineResult
from app.models.task import DeliveryConfig, DeliveryType, Task, TaskStatus


# ── Shared constants ──────────────────────────────────────────────────────────

_TASK_ID      = "task-001"
_COND_ID      = "cond.churn_alert"
_COND_V1      = "1.0"
_COND_V2      = "1.1"
_CONCEPT_ID   = "org.churn_risk_score"
_CONCEPT_V    = "1.0"
_ACTION_ID    = "action.slack_alert"
_ACTION_V     = "1.0"
_ENTITY       = "org_123"
_TOKEN        = "calib-token-abc123"
_INTENT       = "alert when churn risk score exceeds 0.7"
_TS           = "2024-01-15T12:00:00Z"
_DELIVERY     = DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://example.com/hook")


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _task(cv: str) -> Task:
    return Task(
        task_id=_TASK_ID, intent=_INTENT,
        concept_id=_CONCEPT_ID, concept_version=_CONCEPT_V,
        condition_id=_COND_ID, condition_version=cv,
        action_id=_ACTION_ID, action_version=_ACTION_V,
        entity_scope=_ENTITY, delivery=_DELIVERY,
        status=TaskStatus.ACTIVE,
    )


def _pipeline(cv: str) -> FullPipelineResult:
    return FullPipelineResult(
        result=ConceptResult(
            value=0.82, type=ConceptOutputType.FLOAT,
            entity=_ENTITY, version=_CONCEPT_V,
            deterministic=True, timestamp=_TS,
        ),
        decision=DecisionResult(
            value=True, type=DecisionType.BOOLEAN,
            entity=_ENTITY, condition_id=_COND_ID,
            condition_version=cv, timestamp=_TS,
        ),
        entity=_ENTITY, timestamp=_TS,
    )


class _TaskSvc:
    async def create_task(self, req: Any) -> Task:   return _task(_COND_V1)
    async def update_task(self, tid: str, b: Any) -> Task: return _task(_COND_V2)


class _ExecuteSvc:
    async def evaluate_full(self, req: Any) -> FullPipelineResult:
        return _pipeline(getattr(req, "condition_version", _COND_V1))


class _FeedbackSvc:
    async def submit(self, req: Any) -> FeedbackResponse:
        return FeedbackResponse(status="recorded", feedback_id="fb-001")


class _CalibSvc:
    async def calibrate(self, req: Any) -> CalibrationResult:
        return CalibrationResult(
            status=CalibrationStatus.RECOMMENDATION_AVAILABLE,
            current_params={"value": 0.70},
            recommended_params={"value": 0.77},
            calibration_token=_TOKEN,
            impact=CalibrationImpact(delta_alerts=-1.0, direction=ImpactDirection.DECREASE),
        )

    async def apply_calibration(self, req: Any) -> ApplyCalibrationResult:
        return ApplyCalibrationResult(
            condition_id=_COND_ID,
            previous_version=_COND_V1, new_version=_COND_V2,
            params_applied={"value": 0.77},
            tasks_pending_rebind=[TaskPendingRebind(task_id=_TASK_ID, intent=_INTENT)],
        )


# ── Build test app ────────────────────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=_null_lifespan)
app.add_exception_handler(MemintelError, memintel_error_handler)
app.include_router(tasks_route.router)
app.include_router(execute_route.evaluate_router, prefix="/evaluate")
app.include_router(execute_route.router,          prefix="/execute")
app.include_router(feedback_route.router)
app.include_router(conditions_route.router)

app.dependency_overrides[get_task_authoring_service] = lambda: _TaskSvc()
app.dependency_overrides[get_execute_service]        = lambda: _ExecuteSvc()
app.dependency_overrides[get_feedback_service]       = lambda: _FeedbackSvc()
app.dependency_overrides[get_calibration_service]    = lambda: _CalibSvc()


# ── Uvicorn runner ────────────────────────────────────────────────────────────

def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",   # suppress request logs from uvicorn
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait until the socket is accepting connections (max 5 s).
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not start on port {port} within 5 seconds")


# ── Reporting helpers ─────────────────────────────────────────────────────────

_PASS = "PASS"
_FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []   # (step, status, detail)


def _check(
    step: str,
    label: str,
    r: httpx.Response,
    *,
    expected_status: int = 200,
    required_keys: list[str] | None = None,
    key_values: dict[str, Any] | None = None,
) -> dict:
    """
    Validate a response and record pass/fail.
    Returns the parsed JSON body (or {} on error).
    """
    issues: list[str] = []

    if r.status_code != expected_status:
        issues.append(f"status {r.status_code} != {expected_status}")

    body: dict = {}
    try:
        body = r.json()
    except Exception:
        issues.append("response body is not valid JSON")

    if not issues and required_keys:
        for k in required_keys:
            if k not in body:
                issues.append(f"key '{k}' missing from response")

    if not issues and key_values:
        for k, expected in key_values.items():
            actual = body.get(k)
            if actual != expected:
                issues.append(f"{k}={actual!r} != {expected!r}")

    if issues:
        detail = "; ".join(issues)
        _results.append((_FAIL, step, f"{label} -- {detail}"))
        print(f"  FAIL  {step}: {label}")
        print(f"        {detail}")
        if body:
            import json
            print(f"        body: {json.dumps(body, indent=None)[:300]}")
    else:
        _results.append((_PASS, step, label))
        print(f"  PASS  {step}: {label}")

    return body


# ── Main: 7-step happy path ───────────────────────────────────────────────────

def run_happy_path(base: str) -> None:
    print(f"\nTarget: {base}\n")

    with httpx.Client(base_url=base, timeout=10) as client:

        # ── Step 1: POST /tasks ───────────────────────────────────────────────
        r1 = client.post("/tasks", json={
            "intent":       _INTENT,
            "entity_scope": _ENTITY,
            "delivery":     {"type": "webhook", "endpoint": "https://example.com/hook"},
        })
        b1 = _check(
            "Step 1", "POST /tasks -> 200, task_id present, condition_version=1.0",
            r1,
            required_keys=["task_id", "condition_version", "concept_id"],
            key_values={"condition_version": _COND_V1},
        )
        task_id = b1.get("task_id", _TASK_ID)

        # ── Step 2: POST /evaluate/full (v1) ─────────────────────────────────
        r2 = client.post("/evaluate/full", json={
            "concept_id":        _CONCEPT_ID,
            "concept_version":   _CONCEPT_V,
            "condition_id":      _COND_ID,
            "condition_version": _COND_V1,
            "entity":            _ENTITY,
        })
        b2 = _check(
            "Step 2", "POST /evaluate/full -> 200, result + decision present",
            r2,
            required_keys=["result", "decision", "entity"],
        )
        # Structural sub-checks for FullPipelineResult
        if b2 and "result" in b2 and "decision" in b2:
            r_node, d_node = b2["result"], b2["decision"]
            sub_issues = []
            if "value" not in r_node:
                sub_issues.append("result.value missing")
            if "actions_triggered" not in d_node:
                sub_issues.append("decision.actions_triggered missing (must be nested in decision, not top-level)")
            if d_node.get("condition_version") != _COND_V1:
                sub_issues.append(f"decision.condition_version={d_node.get('condition_version')!r} != {_COND_V1!r}")
            if sub_issues:
                _results.append((_FAIL, "Step 2", "; ".join(sub_issues)))
                print(f"  FAIL  Step 2 sub-checks: {'; '.join(sub_issues)}")
            else:
                _results.append((_PASS, "Step 2", "FullPipelineResult structure valid"))
                print(f"  PASS  Step 2: FullPipelineResult shape correct (actions_triggered nested in decision)")

        # ── Step 3: POST /feedback/decision ──────────────────────────────────
        # Spec path: /feedback/decision  (developer_api.yaml line 1243)
        r3 = client.post("/feedback/decision", json={
            "condition_id":      _COND_ID,
            "condition_version": _COND_V1,
            "entity":            _ENTITY,
            "timestamp":         _TS,
            "feedback":          "false_positive",
        })
        _check(
            "Step 3", "POST /feedback/decision -> 200, status=recorded",
            r3,
            required_keys=["status", "feedback_id"],
            key_values={"status": "recorded"},
        )

        # ── Step 4: POST /conditions/calibrate ───────────────────────────────
        r4 = client.post("/conditions/calibrate", json={
            "condition_id":       _COND_ID,
            "condition_version":  _COND_V1,
            "feedback_direction": "tighten",
        })
        b4 = _check(
            "Step 4", "POST /conditions/calibrate -> 200, recommendation_available + token",
            r4,
            required_keys=["status", "calibration_token", "recommended_params"],
            key_values={"status": "recommendation_available"},
        )
        calibration_token = b4.get("calibration_token", _TOKEN)

        # ── Step 5: POST /conditions/apply-calibration ───────────────────────
        # Spec path (developer_api.yaml line 1154): /conditions/apply-calibration
        # NOTE: the user's prompt uses /conditions/calibrate/apply -- that path
        #       does NOT exist in the spec or the implementation.
        r5 = client.post("/conditions/apply-calibration", json={
            "calibration_token": calibration_token,
        })
        b5 = _check(
            "Step 5", "POST /conditions/apply-calibration -> 200, new_version=1.1",
            r5,
            required_keys=["condition_id", "new_version", "tasks_pending_rebind"],
            key_values={"new_version": _COND_V2},
        )
        new_version = b5.get("new_version", _COND_V2)

        # Verify the spec note that tasks_pending_rebind is informational
        if b5 and isinstance(b5.get("tasks_pending_rebind"), list):
            pending = b5["tasks_pending_rebind"]
            if pending and pending[0].get("task_id") == _TASK_ID:
                _results.append((_PASS, "Step 5", "tasks_pending_rebind contains the task"))
                print(f"  PASS  Step 5: tasks_pending_rebind[0].task_id={_TASK_ID!r} (informational)")
            else:
                _results.append((_PASS, "Step 5", "tasks_pending_rebind returned (empty or different task)"))

        # ── Step 6: PATCH /tasks/{id} ─────────────────────────────────────────
        r6 = client.patch(f"/tasks/{task_id}", json={"condition_version": new_version})
        _check(
            "Step 6", f"PATCH /tasks/{task_id} -> 200, condition_version=1.1",
            r6,
            required_keys=["task_id", "condition_version"],
            key_values={"condition_version": _COND_V2},
        )

        # ── Step 7: POST /evaluate/full again (v2) ────────────────────────────
        r7 = client.post("/evaluate/full", json={
            "concept_id":        _CONCEPT_ID,
            "concept_version":   _CONCEPT_V,
            "condition_id":      _COND_ID,
            "condition_version": _COND_V2,
            "entity":            _ENTITY,
        })
        b7 = _check(
            "Step 7", "POST /evaluate/full -> 200, decision.condition_version=1.1 confirmed",
            r7,
            required_keys=["result", "decision"],
        )
        if b7 and "decision" in b7:
            cv_got = b7["decision"].get("condition_version")
            if cv_got == _COND_V2:
                _results.append((_PASS, "Step 7", f"New condition version {_COND_V2} confirmed in response"))
                print(f"  PASS  Step 7: decision.condition_version={_COND_V2!r} -- rebind confirmed")
            else:
                _results.append((_FAIL, "Step 7", f"decision.condition_version={cv_got!r} != {_COND_V2!r}"))
                print(f"  FAIL  Step 7: decision.condition_version={cv_got!r} != {_COND_V2!r}")

        # ── Spec discrepancy notes ─────────────────────────────────────────────
        print()
        print("Spec notes (developer_api.yaml):")
        print("  • Step 3 path: /feedback/decision  (not /feedback -- no bare endpoint defined)")
        print("  • Step 5 path: /conditions/apply-calibration  (spec line 1154)")
        print("    The prompt used /conditions/calibrate/apply -- that path is NOT in the spec")
        print("    or the implementation.  Correct path: /conditions/apply-calibration")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    port = _free_port()
    print(f"Starting Memintel test app on 127.0.0.1:{port} …", end="", flush=True)
    server = _start_server(port)
    print(" ready.")

    try:
        run_happy_path(f"http://127.0.0.1:{port}")
    finally:
        server.should_exit = True

    print()
    passes = sum(1 for s, _, _ in _results if s == _PASS)
    fails  = sum(1 for s, _, _ in _results if s == _FAIL)
    total  = len(_results)

    print(f"{'─'*60}")
    print(f"Results: {passes}/{total} checks passed, {fails} failed")
    print(f"{'─'*60}")

    if fails:
        print("\nFailed checks:")
        for status, step, detail in _results:
            if status == _FAIL:
                print(f"  [{step}] {detail}")
        return 1

    print("\nAll routes returned expected status codes and response shapes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
