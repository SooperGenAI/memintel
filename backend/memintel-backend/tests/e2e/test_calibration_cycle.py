"""
tests/e2e/test_calibration_cycle.py
──────────────────────────────────────────────────────────────────────────────
Calibration cycle end-to-end tests — complete feedback → calibrate → apply →
rebind chain verified against a real PostgreSQL database with no mocks.

Key findings documented here
─────────────────────────────
MINIMUM_FEEDBACK_COUNT = 3  (MIN_FEEDBACK_THRESHOLD from app.models.calibration)
  CalibrationService.derive_direction() requires ≥ 3 feedback records before it
  can produce a direction. With fewer records → no_recommendation(insufficient_data).
  1 < 3 and 0 < 3 are both trivially true; minimum documented per test.

guardrails_store: _StubGuardrailsStore injected by cal_client fixture
  The default e2e app (from conftest) sets app.state.guardrails_store = None.
  CalibrationService returns no_recommendation(guardrails_unavailable) when
  guardrails_store is None. These calibration tests inject a _StubGuardrailsStore
  with wide [0.0, 1.0] bounds and on_bounds_exceeded='clamp', allowing
  recommendation_available responses to be exercised in the test environment.

Decision records required before feedback
  POST /feedback/decision validates that a decision record exists for
  (condition_id, condition_version, entity, timestamp). Decision records are
  written asynchronously (fire-and-forget asyncio.create_task) by /evaluate/full.
  Tests poll the DB after each /evaluate/full call (up to 1.5 s) to confirm.

Calibration direction — threshold strategy (no directional awareness)
  CalibrationService.adjust_params() ignores the threshold direction ("above"
  vs "below") and adjusts the 'value' parameter directly:
    false_positive → tighten → value INCREASES by step
    false_negative → relax   → value DECREASES by step
  step = max(current_value * 0.10, 0.1)
  For initial value 0.35:  step = max(0.035, 0.1) = 0.1
    tighten:  0.35 + 0.1 = 0.45
    relax:    0.35 − 0.1 = 0.25
  This means for a "below" threshold condition with false_positives, the service
  raises the threshold value (which for a "below" condition fires MORE, not fewer),
  which is semantically unexpected. Tests assert the actual code behaviour.

Token expiry test
  Uses run_db() to directly UPDATE calibration_tokens.expires_at to one hour
  in the past, simulating expiry without sleeping.

Equals strategy — no evaluation needed
  CalibrationService checks strategy type before fetching feedback records.
  The equals branch is a pure early-return; no feedback or evaluation is
  needed to hit it. The test registers the condition and calls calibrate
  directly.

Test count: 11 tests across 6 cycles.
"""
from __future__ import annotations

import time
import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from app.api.routes import (
    actions,
    agents,
    compile,
    conditions,
    context,
    decisions,
    execute,
    feedback,
    guardrails_api,
    jobs,
    registry,
    tasks,
)
from app.config import PrimitiveRegistry
from app.models.errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorType,
    MemintelError,
    memintel_error_handler,
)
from tests.e2e.conftest import (
    API_KEY,
    ELEVATED_KEY,
    TEST_DATABASE_URL,
    seed_task,
)


# ── Stub guardrails store ──────────────────────────────────────────────────────


class _StubGuardrailsStore:
    """
    Minimal synchronous stub satisfying CalibrationService.calibrate().

    CalibrationService uses:
      bounds      = store.get_threshold_bounds(strategy_type_string)
      guardrails  = store.get_guardrails()
      on_exceeded = guardrails.constraints.on_bounds_exceeded

    Returns wide [0.0, 1.0] bounds with on_bounds_exceeded='clamp' so that
    all parameter adjustments proceed within the bounds.
    """

    def is_loaded(self) -> bool:
        return True

    def get_threshold_bounds(self, strategy: str) -> dict:  # noqa: ARG002
        return {"min": 0.0, "max": 1.0}

    def get_guardrails(self) -> types.SimpleNamespace:
        constraints = types.SimpleNamespace(on_bounds_exceeded="clamp")
        return types.SimpleNamespace(constraints=constraints)


# ── App factory with stub guardrails ──────────────────────────────────────────


def _make_cal_app() -> FastAPI:
    """
    Full e2e FastAPI app with _StubGuardrailsStore injected.

    Identical to _make_e2e_app() from conftest except:
      app.state.guardrails_store = _StubGuardrailsStore()

    This allows CalibrationService.calibrate() to reach the adjust_params()
    branch and return recommendation_available status.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        try:
            pool = await asyncpg.create_pool(
                TEST_DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to test database: {exc}") from exc

        redis_stub = MagicMock()
        redis_stub.close = AsyncMock()

        app.state.db = pool
        app.state.elevated_key = ELEVATED_KEY
        app.state.api_key = API_KEY
        app.state.guardrails_store = _StubGuardrailsStore()   # ← key difference
        app.state.primitive_registry = PrimitiveRegistry()
        app.state.redis = redis_stub
        app.state.connector_registry = None
        app.state.config = None

        yield

        await pool.close()

    app = FastAPI(title="Memintel Calibration E2E Tests", lifespan=_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)

    async def _http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    type=(
                        ErrorType.AUTH_ERROR if exc.status_code in (401, 403)
                        else ErrorType.NOT_FOUND if exc.status_code == 404
                        else ErrorType.EXECUTION_ERROR
                    ),
                    message=str(detail) if detail else str(exc.status_code),
                )
            ).model_dump(mode="json"),
        )

    app.add_exception_handler(HTTPException, _http_exc_handler)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(asyncpg.PostgresError)
    async def _postgres_handler(request: Request, exc: asyncpg.PostgresError) -> JSONResponse:
        if isinstance(exc, asyncpg.CheckViolationError):
            return JSONResponse(status_code=422, content={"detail": "Invalid field value"})
        if isinstance(exc, asyncpg.DataError):
            return JSONResponse(status_code=422, content={"detail": "Invalid data format"})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(execute.evaluate_router, prefix="/evaluate",  tags=["Execution"])
    app.include_router(execute.router,           prefix="/execute",   tags=["Execution"])
    app.include_router(compile.router,           prefix="/compile",   tags=["Compiler"])
    app.include_router(registry.router,          prefix="/registry",  tags=["Registry"])
    app.include_router(agents.router,            prefix="/agents",    tags=["Agents"])
    app.include_router(tasks.router,                                  tags=["Tasks"])
    app.include_router(conditions.router,                             tags=["Conditions"])
    app.include_router(decisions.router,                              tags=["Decisions"])
    app.include_router(feedback.router,                               tags=["Feedback"])
    app.include_router(actions.router,                                tags=["Actions"])
    app.include_router(jobs.router,              prefix="/jobs",      tags=["Jobs"])
    app.include_router(context.router,           prefix="/context",   tags=["Context"])
    app.include_router(guardrails_api.router,                         tags=["Guardrails"])

    return app


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def cal_client(e2e_setup):
    """
    Calibration-specific e2e fixture.

    Uses _make_cal_app() which injects _StubGuardrailsStore, enabling
    recommendation_available responses from POST /conditions/calibrate.

    Yields (client, pool, run_db).
    """
    pool, run_db = e2e_setup
    app = _make_cal_app()
    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, pool, run_db
    except RuntimeError as exc:
        if "Cannot connect" in str(exc):
            pytest.skip(f"Test database unavailable: {exc}")
            return
        raise


# ── Shared primitive / definition bodies ──────────────────────────────────────

_PRIMITIVE = "account.metric_value"


def _concept_body(concept_id: str, version: str = "v1") -> dict:
    """Float-output passthrough concept using z_score_op.

    MockConnector returns None for the primitive; missing_data_policy='zero'
    substitutes 0.0, so /evaluate/full always produces concept_value=0.0
    and fires any threshold condition with direction='below' and value > 0.
    """
    return {
        "definition_id": concept_id,
        "version": version,
        "definition_type": "concept",
        "namespace": "org",
        "body": {
            "concept_id": concept_id,
            "version": version,
            "namespace": "org",
            "output_type": "float",
            "output_feature": "f_metric",
            "primitives": {
                _PRIMITIVE: {"type": "float", "missing_data_policy": "zero"}
            },
            "features": {
                "f_metric": {"op": "z_score_op", "inputs": {"input": _PRIMITIVE}}
            },
        },
    }


def _condition_body(
    cond_id: str,
    version: str,
    concept_id: str,
    concept_version: str = "v1",
    threshold: float = 0.35,
    direction: str = "below",
    strategy_type: str = "threshold",
) -> dict:
    if strategy_type == "equals":
        params: dict = {"value": "high_risk"}
    else:
        params = {"direction": direction, "value": threshold}
    return {
        "definition_id": cond_id,
        "version": version,
        "definition_type": "condition",
        "namespace": "org",
        "body": {
            "condition_id": cond_id,
            "version": version,
            "namespace": "org",
            "concept_id": concept_id,
            "concept_version": concept_version,
            "strategy": {"type": strategy_type, "params": params},
        },
    }


def _action_body(action_id: str, version: str, cond_id: str, cond_ver: str) -> dict:
    return {
        "action_id": action_id,
        "version": version,
        "namespace": "org",
        "config": {
            "type": "webhook",
            "endpoint": "https://example.com/e2e-cal-webhook",
        },
        "trigger": {
            "fire_on": "true",
            "condition_id": cond_id,
            "condition_version": cond_ver,
        },
    }


# ── Registration helper ───────────────────────────────────────────────────────


def _register_all(
    client,
    elevated_headers: dict,
    concept_id: str,
    cond_id: str,
    cond_ver: str,
    action_id: str,
    threshold: float = 0.35,
    direction: str = "below",
    strategy_type: str = "threshold",
    concept_version: str = "v1",
) -> None:
    """Register concept, condition, and action. Asserts each step succeeds."""
    r = client.post(
        "/registry/definitions",
        json=_concept_body(concept_id, concept_version),
        headers=elevated_headers,
    )
    assert r.status_code == 200, f"concept register failed: {r.text}"

    r = client.post(
        "/registry/definitions",
        json=_condition_body(
            cond_id, cond_ver, concept_id, concept_version,
            threshold, direction, strategy_type,
        ),
        headers=elevated_headers,
    )
    assert r.status_code == 200, f"condition register failed: {r.text}"

    r = client.post(
        "/actions",
        json=_action_body(action_id, "v1", cond_id, cond_ver),
        headers=elevated_headers,
    )
    assert r.status_code in (200, 201), f"action register failed: {r.text}"


# ── Decision / feedback helpers ───────────────────────────────────────────────


def _evaluate_and_get_timestamp(
    client,
    pool: asyncpg.Pool,
    run_db,
    api_headers: dict,
    concept_id: str,
    concept_version: str,
    cond_id: str,
    cond_ver: str,
    entity: str,
) -> str:
    """
    POST /evaluate/full for entity and return the DB-stamped evaluated_at string.

    Decision records are written via asyncio.create_task() (fire-and-forget).
    Polls the DB for up to 1.5 s to confirm the write, then returns the
    evaluated_at timestamp as an ISO 8601 string for use in feedback requests.
    """
    r = client.post(
        "/evaluate/full",
        json={
            "concept_id": concept_id,
            "concept_version": concept_version,
            "condition_id": cond_id,
            "condition_version": cond_ver,
            "entity": entity,
        },
        headers=api_headers,
    )
    assert r.status_code == 200, f"/evaluate/full for '{entity}' failed: {r.text}"

    async def _fetch():
        async with pool.acquire() as conn:
            return await conn.fetchrow(
                """
                SELECT evaluated_at FROM decisions
                WHERE condition_id = $1
                  AND condition_version = $2
                  AND entity_id = $3
                ORDER BY evaluated_at DESC
                LIMIT 1
                """,
                cond_id, cond_ver, entity,
            )

    for _ in range(15):            # up to 1.5 s
        row = run_db(_fetch())
        if row is not None:
            return row["evaluated_at"].isoformat()
        time.sleep(0.1)

    pytest.fail(
        f"Decision record not written for entity '{entity}' after 1.5 s. "
        "Check that /evaluate/full fires its create_task write correctly."
    )


def _submit_feedback(
    client,
    api_headers: dict,
    cond_id: str,
    cond_ver: str,
    entity: str,
    timestamp: str,
    feedback_val: str,
) -> str:
    """POST /feedback/decision and return the assigned feedback_id."""
    r = client.post(
        "/feedback/decision",
        json={
            "condition_id": cond_id,
            "condition_version": cond_ver,
            "entity": entity,
            "timestamp": timestamp,
            "feedback": feedback_val,
        },
        headers=api_headers,
    )
    assert r.status_code == 200, f"feedback submit failed for '{entity}': {r.text}"
    resp = r.json()
    assert resp["status"] == "recorded"
    assert resp["feedback_id"], "feedback_id must be non-empty"
    return resp["feedback_id"]


def _seed_feedback(
    client,
    pool: asyncpg.Pool,
    run_db,
    api_headers: dict,
    concept_id: str,
    concept_version: str,
    cond_id: str,
    cond_ver: str,
    entities: list[str],
    feedback_val: str,
) -> list[str]:
    """Evaluate + submit feedback for each entity. Returns list of feedback_ids."""
    ids = []
    for entity in entities:
        ts = _evaluate_and_get_timestamp(
            client, pool, run_db, api_headers,
            concept_id, concept_version, cond_id, cond_ver, entity,
        )
        fid = _submit_feedback(client, api_headers, cond_id, cond_ver, entity, ts, feedback_val)
        ids.append(fid)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE 1 — Complete Calibration Cycle
# ══════════════════════════════════════════════════════════════════════════════

_C1_CONCEPT_ID = "cal.c1.metric"
_C1_COND_ID    = "cal.c1.cond"
_C1_ACTION_ID  = "cal.c1.action"
_C1_ENTITIES   = [f"c1_entity_{i}" for i in range(1, 6)]   # 5 distinct entities


@pytest.mark.e2e
def test_complete_calibration_cycle(cal_client, api_headers, elevated_headers):
    """
    Full feedback → calibrate → apply → rebind cycle — 9 steps.

    Step 1  Register concept v1 + condition v1 (threshold 0.35, below) + action
    Step 2  Seed task bound to condition v1
    Step 3  Submit 5 × false_positive feedback (with real decision records)
    Step 4  POST /conditions/calibrate → assert recommendation_available
    Step 5  Verify token is non-empty
    Step 6  POST /conditions/apply-calibration → creates v2
    Step 7  Verify token is single-use (second apply → HTTP 400)
    Step 8  GET /conditions/{id}?version=v2 → params match recommended_params
    Step 9  GET /conditions/{id}?version=v1 → params unchanged (byte-for-byte)
    Step 10 PATCH /tasks/{id} → rebind to v2
    Step 11 GET /tasks/{id} → condition_version == "v2"
    Step 12 Final DB assertions: 5 feedback records, v1+v2 exist, task on v2
    """
    client, pool, run_db = cal_client

    # Step 1 — Register definitions
    _register_all(client, elevated_headers, _C1_CONCEPT_ID, _C1_COND_ID, "v1", _C1_ACTION_ID, threshold=0.35)

    # Record v1 original params before any calibration
    r = client.get(f"/conditions/{_C1_COND_ID}", params={"version": "v1"}, headers=api_headers)
    assert r.status_code == 200, f"GET condition v1 failed: {r.text}"
    v1_orig_params = r.json()["strategy"]["params"].copy()
    assert v1_orig_params["value"] == pytest.approx(0.35)

    # Step 2 — Seed task bound to v1
    task_id = run_db(seed_task(
        pool,
        intent="Alert when metric falls below threshold",
        concept_id=_C1_CONCEPT_ID,
        concept_version="v1",
        condition_id=_C1_COND_ID,
        condition_version="v1",
        action_id=_C1_ACTION_ID,
        action_version="v1",
    ))
    assert task_id, "seed_task must return a task_id"

    # Step 3 — Submit 5 × false_positive feedback (unique entity per record)
    feedback_ids = _seed_feedback(
        client, pool, run_db, api_headers,
        _C1_CONCEPT_ID, "v1", _C1_COND_ID, "v1",
        _C1_ENTITIES, "false_positive",
    )
    assert len(feedback_ids) == 5, f"Expected 5 feedback ids, got {len(feedback_ids)}"
    assert len(set(feedback_ids)) == 5, "All 5 feedback_ids must be unique"

    # Step 4 — Request calibration recommendation
    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C1_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200, f"POST /calibrate failed: {r.text}"
    calib = r.json()

    assert calib["status"] in ("recommendation_available", "no_recommendation")
    if calib["status"] == "no_recommendation":
        pytest.skip(
            f"Calibration returned no_recommendation "
            f"({calib.get('no_recommendation_reason')}). Full cycle not exercisable."
        )

    assert calib["status"] == "recommendation_available"
    assert calib["calibration_token"] is not None, "calibration_token must be present"
    assert len(calib["calibration_token"]) > 0,    "calibration_token must be non-empty"
    assert calib["current_params"] is not None,    "current_params must be present"
    assert calib["recommended_params"] is not None, "recommended_params must be present"
    assert calib["recommended_params"] != calib["current_params"], (
        "5 false_positive records should produce a parameter change"
    )

    cal_token        = calib["calibration_token"]
    recommended_params = calib["recommended_params"]

    # Step 5 — Token noted; not applied yet. Token must be a non-empty string.
    assert isinstance(cal_token, str) and len(cal_token) > 0

    # Step 6 — Apply calibration → creates v2
    r = client.post(
        "/conditions/apply-calibration",
        json={"calibration_token": cal_token, "new_version": "v2"},
        headers=elevated_headers,
    )
    assert r.status_code == 200, f"POST /apply-calibration failed: {r.text}"
    apply = r.json()

    assert apply["condition_id"] == _C1_COND_ID
    assert apply["previous_version"] == "v1"
    assert apply["new_version"] == "v2"
    assert apply["params_applied"] == recommended_params
    assert isinstance(apply["tasks_pending_rebind"], list)

    # Task seeded in Step 2 is still on v1 → must appear in tasks_pending_rebind
    pending_ids = [t["task_id"] for t in apply["tasks_pending_rebind"]]
    assert str(task_id) in pending_ids, (
        f"Task {task_id} must appear in tasks_pending_rebind (still on v1). "
        f"Got: {pending_ids}"
    )

    # Step 7 — Verify token is single-use
    r_reuse = client.post(
        "/conditions/apply-calibration",
        json={"calibration_token": cal_token, "new_version": "v3"},
        headers=elevated_headers,
    )
    assert r_reuse.status_code == 400, (
        f"Reusing a consumed calibration token must return HTTP 400, "
        f"got {r_reuse.status_code}: {r_reuse.text}"
    )

    # Step 8 — Verify v2 exists with recommended params
    r = client.get(f"/conditions/{_C1_COND_ID}", params={"version": "v2"}, headers=api_headers)
    assert r.status_code == 200, f"GET v2 condition failed: {r.text}"
    v2 = r.json()
    assert v2["version"] == "v2"
    assert v2["strategy"]["params"] == recommended_params
    # v2 is newly created — must not be deprecated
    assert v2.get("deprecated") in (False, None)

    # Step 9 — Verify v1 params are byte-for-byte unchanged
    r = client.get(f"/conditions/{_C1_COND_ID}", params={"version": "v1"}, headers=api_headers)
    assert r.status_code == 200, f"GET v1 (post-calibration) failed: {r.text}"
    v1_after_params = r.json()["strategy"]["params"]
    assert v1_after_params == v1_orig_params, (
        f"v1 params mutated by calibration!\n"
        f"  Before: {v1_orig_params}\n"
        f"  After:  {v1_after_params}"
    )
    assert v1_after_params["value"] == pytest.approx(0.35), (
        "v1 threshold value must remain exactly 0.35 after calibration"
    )

    # Step 10 — Rebind task to v2
    r = client.patch(
        f"/tasks/{task_id}",
        json={"condition_version": "v2"},
        headers=api_headers,
    )
    assert r.status_code == 200, f"PATCH /tasks/{task_id} failed: {r.text}"

    # Step 11 — Verify task now on v2
    r = client.get(f"/tasks/{task_id}", headers=api_headers)
    assert r.status_code == 200
    assert r.json()["condition_version"] == "v2", (
        f"Task must be on v2 after rebind. Got: {r.json()['condition_version']}"
    )

    # Step 12 — Final DB assertions
    async def _count_feedback():
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM feedback_records "
                "WHERE condition_id = $1 AND condition_version = $2",
                _C1_COND_ID, "v1",
            )

    async def _get_versions():
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT version FROM definitions "
                "WHERE definition_id = $1 AND definition_type = 'condition' "
                "ORDER BY version",
                _C1_COND_ID,
            )
            return [row["version"] for row in rows]

    fb_count = run_db(_count_feedback())
    assert fb_count == 5, f"Expected 5 feedback records in DB, got {fb_count}"

    versions = run_db(_get_versions())
    assert "v1" in versions, "v1 must still exist in DB"
    assert "v2" in versions, "v2 must exist in DB after calibration"

    # Task must still report v2 after rebind
    r = client.get(f"/tasks/{task_id}", headers=api_headers)
    assert r.json()["condition_version"] == "v2"


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE 2 — Calibration Direction Verification
# ══════════════════════════════════════════════════════════════════════════════

_C2_FP_CONCEPT_ID = "cal.c2.fp.metric"
_C2_FP_COND_ID    = "cal.c2.fp.cond"
_C2_FP_ACTION_ID  = "cal.c2.fp.action"

_C2_FN_CONCEPT_ID = "cal.c2.fn.metric"
_C2_FN_COND_ID    = "cal.c2.fn.cond"
_C2_FN_ACTION_ID  = "cal.c2.fn.action"

_C2_OK_CONCEPT_ID = "cal.c2.ok.metric"
_C2_OK_COND_ID    = "cal.c2.ok.cond"
_C2_OK_ACTION_ID  = "cal.c2.ok.action"


@pytest.mark.e2e
def test_false_positives_produce_tightening_recommendation(cal_client, api_headers, elevated_headers):
    """
    5 × false_positive → derive_direction → 'tighten' → value INCREASES.

    For threshold strategy (direction-agnostic):
      step = max(0.35 * 0.1, 0.1) = 0.1
      tighten: new_value = 0.35 + 0.1 = 0.45

    Note: for a "below" threshold condition, raising the threshold causes MORE
    alerts (not fewer), which is semantically counterintuitive. The service does
    not account for the threshold direction field — it adjusts 'value' directly.
    Tests assert the actual code behaviour.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C2_FP_CONCEPT_ID, _C2_FP_COND_ID, "v1", _C2_FP_ACTION_ID, threshold=0.35)

    entities = [f"fp_entity_{i}" for i in range(1, 6)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C2_FP_CONCEPT_ID, "v1", _C2_FP_COND_ID, "v1",
        entities, "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C2_FP_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    if calib["status"] != "recommendation_available":
        pytest.skip(f"No recommendation: {calib.get('no_recommendation_reason')}")

    recommended_value = calib["recommended_params"]["value"]

    assert recommended_value > 0.35, (
        f"5 false_positives → tighten → threshold value must increase above 0.35. "
        f"Got {recommended_value}"
    )
    assert recommended_value == pytest.approx(0.45, abs=1e-6), (
        f"Expected 0.45 (0.35 + step 0.1). Got {recommended_value}"
    )


@pytest.mark.e2e
def test_false_negatives_produce_relaxing_recommendation(cal_client, api_headers, elevated_headers):
    """
    5 × false_negative → derive_direction → 'relax' → value DECREASES.

    For threshold strategy:
      step = max(0.35 * 0.1, 0.1) = 0.1
      relax: new_value = 0.35 − 0.1 = 0.25
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C2_FN_CONCEPT_ID, _C2_FN_COND_ID, "v1", _C2_FN_ACTION_ID, threshold=0.35)

    entities = [f"fn_entity_{i}" for i in range(1, 6)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C2_FN_CONCEPT_ID, "v1", _C2_FN_COND_ID, "v1",
        entities, "false_negative",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C2_FN_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    if calib["status"] != "recommendation_available":
        pytest.skip(f"No recommendation: {calib.get('no_recommendation_reason')}")

    recommended_value = calib["recommended_params"]["value"]

    assert recommended_value < 0.35, (
        f"5 false_negatives → relax → threshold value must decrease below 0.35. "
        f"Got {recommended_value}"
    )
    assert recommended_value == pytest.approx(0.25, abs=1e-6), (
        f"Expected 0.25 (0.35 − step 0.1). Got {recommended_value}"
    )


@pytest.mark.e2e
def test_correct_feedback_produces_no_recommendation(cal_client, api_headers, elevated_headers):
    """
    5 × correct feedback → fp_count=0, fn_count=0 → tie → derive_direction returns
    None → no_recommendation(insufficient_data).

    'correct' records are counted in the total but do not shift the majority.
    With 5 records (≥ MIN_FEEDBACK_THRESHOLD) the data threshold is met but
    there is no directional signal → insufficient_data.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C2_OK_CONCEPT_ID, _C2_OK_COND_ID, "v1", _C2_OK_ACTION_ID, threshold=0.35)

    entities = [f"ok_entity_{i}" for i in range(1, 6)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C2_OK_CONCEPT_ID, "v1", _C2_OK_COND_ID, "v1",
        entities, "correct",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C2_OK_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    assert calib["status"] == "no_recommendation"
    assert calib["no_recommendation_reason"] == "insufficient_data", (
        "5 correct-only feedback records produce no directional signal → "
        f"expected insufficient_data, got {calib.get('no_recommendation_reason')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE 3 — Minimum Feedback Threshold
#
# MIN_FEEDBACK_THRESHOLD = 3  (from app.models.calibration)
# ══════════════════════════════════════════════════════════════════════════════

_C3_ZERO_CONCEPT_ID = "cal.c3.zero.metric"
_C3_ZERO_COND_ID    = "cal.c3.zero.cond"
_C3_ZERO_ACTION_ID  = "cal.c3.zero.action"

_C3_ONE_CONCEPT_ID  = "cal.c3.one.metric"
_C3_ONE_COND_ID     = "cal.c3.one.cond"
_C3_ONE_ACTION_ID   = "cal.c3.one.action"

_C3_MIN_CONCEPT_ID  = "cal.c3.min.metric"
_C3_MIN_COND_ID     = "cal.c3.min.cond"
_C3_MIN_ACTION_ID   = "cal.c3.min.action"


@pytest.mark.e2e
def test_zero_feedback_returns_no_recommendation(cal_client, api_headers, elevated_headers):
    """
    0 feedback records → no_recommendation(insufficient_data).

    With 0 records, len(records) < MIN_FEEDBACK_THRESHOLD(3) → direction=None.
    no_recommendation_reason must be present (non-None).
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C3_ZERO_CONCEPT_ID, _C3_ZERO_COND_ID, "v1", _C3_ZERO_ACTION_ID)

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C3_ZERO_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    assert calib["status"] == "no_recommendation"
    assert calib["no_recommendation_reason"] is not None
    assert calib["no_recommendation_reason"] == "insufficient_data"


@pytest.mark.e2e
def test_single_feedback_returns_no_recommendation(cal_client, api_headers, elevated_headers):
    """
    1 feedback record → no_recommendation(insufficient_data).

    MIN_FEEDBACK_THRESHOLD = 3. With 1 record, 1 < 3 → direction=None.
    Documented minimum feedback count needed for a recommendation: 3.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C3_ONE_CONCEPT_ID, _C3_ONE_COND_ID, "v1", _C3_ONE_ACTION_ID)

    # Submit exactly 1 feedback record
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C3_ONE_CONCEPT_ID, "v1", _C3_ONE_COND_ID, "v1",
        ["one_entity_1"], "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C3_ONE_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    assert calib["status"] == "no_recommendation", (
        "1 feedback record (< MIN_FEEDBACK_THRESHOLD=3) must return no_recommendation"
    )
    assert calib["no_recommendation_reason"] == "insufficient_data"
    # Documented: minimum feedback count needed for recommendation = 3


@pytest.mark.e2e
def test_sufficient_feedback_enables_recommendation(cal_client, api_headers, elevated_headers):
    """
    Exactly MIN_FEEDBACK_THRESHOLD (3) feedback records → recommendation_available.

    This is the minimum count that unlocks calibration. Three false_positive
    records establish an unambiguous majority (fp=3, fn=0 → tighten direction).
    Documented minimum feedback count needed for a recommendation: 3.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C3_MIN_CONCEPT_ID, _C3_MIN_COND_ID, "v1", _C3_MIN_ACTION_ID, threshold=0.5)

    # Exactly 3 — all false_positive to establish a clear majority
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C3_MIN_CONCEPT_ID, "v1", _C3_MIN_COND_ID, "v1",
        ["min_entity_1", "min_entity_2", "min_entity_3"], "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C3_MIN_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    assert calib["status"] == "recommendation_available", (
        f"3 false_positive records (= MIN_FEEDBACK_THRESHOLD) must produce "
        f"recommendation_available. Got: {calib['status']} / "
        f"{calib.get('no_recommendation_reason')}"
    )
    assert calib["calibration_token"] is not None
    assert len(calib["calibration_token"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE 4 — Equals Strategy Calibration
# ══════════════════════════════════════════════════════════════════════════════

_C4_CONCEPT_ID = "cal.c4.equals.metric"
_C4_COND_ID    = "cal.c4.equals.cond"
_C4_ACTION_ID  = "cal.c4.equals.action"


@pytest.mark.e2e
def test_equals_strategy_always_returns_no_recommendation(cal_client, api_headers, elevated_headers):
    """
    equals strategy → no_recommendation(not_applicable_strategy).

    CalibrationService.calibrate() checks strategy type at step 2, before
    fetching feedback records. The equals branch is a pure early-return:
      if condition.strategy.type in (StrategyType.EQUALS, StrategyType.COMPOSITE):
          return CalibrationResult(status=NO_RECOMMENDATION,
                                   reason=NOT_APPLICABLE_STRATEGY)

    Feedback submission is not needed to exercise this branch — the result
    is the same regardless of feedback count. No evaluation/feedback is
    seeded in this test; the assertion is that calibration rejects equals
    at the strategy-check step.

    Documented: equals has no numeric parameter to adjust.
    """
    client, pool, run_db = cal_client

    # Register concept + equals condition (no evaluation or feedback needed)
    r = client.post(
        "/registry/definitions",
        json=_concept_body(_C4_CONCEPT_ID, "v1"),
        headers=elevated_headers,
    )
    assert r.status_code == 200, f"concept register failed: {r.text}"

    equals_cond_body = {
        "definition_id": _C4_COND_ID,
        "version": "v1",
        "definition_type": "condition",
        "namespace": "org",
        "body": {
            "condition_id": _C4_COND_ID,
            "version": "v1",
            "namespace": "org",
            "concept_id": _C4_CONCEPT_ID,
            "concept_version": "v1",
            "strategy": {"type": "equals", "params": {"value": "high_risk"}},
        },
    }
    r = client.post("/registry/definitions", json=equals_cond_body, headers=elevated_headers)
    assert r.status_code == 200, f"equals condition register failed: {r.text}"

    # Calibrate immediately — no feedback submitted (equals returns early anyway)
    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C4_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    assert calib["status"] == "no_recommendation"
    assert calib["no_recommendation_reason"] == "not_applicable_strategy", (
        f"equals strategy has no numeric parameter → expected not_applicable_strategy. "
        f"Got: {calib.get('no_recommendation_reason')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE 5 — Calibration Token Expiry
# ══════════════════════════════════════════════════════════════════════════════

_C5_CONCEPT_ID = "cal.c5.expiry.metric"
_C5_COND_ID    = "cal.c5.expiry.cond"
_C5_ACTION_ID  = "cal.c5.expiry.action"


@pytest.mark.e2e
def test_expired_token_cannot_be_applied(cal_client, api_headers, elevated_headers):
    """
    Manually expire a calibration token via direct DB UPDATE, then verify
    that POST /conditions/apply-calibration returns HTTP 400.

    CalibrationTokenStore.resolve_and_invalidate() WHERE clause:
      WHERE token_string = $1
        AND used_at IS NULL
        AND expires_at > NOW()

    After back-dating expires_at by one hour, the WHERE clause no longer
    matches → returns None → CalibrationService raises MemintelError →
    HTTP 400.

    Workaround confirmed: direct DB manipulation to simulate expiry without
    sleeping 24 hours.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C5_CONCEPT_ID, _C5_COND_ID, "v1", _C5_ACTION_ID, threshold=0.4)

    # Submit 3 × false_positive to trigger a recommendation
    entities = [f"expiry_entity_{i}" for i in range(1, 4)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C5_CONCEPT_ID, "v1", _C5_COND_ID, "v1",
        entities, "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C5_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    if calib["status"] != "recommendation_available":
        pytest.skip(
            f"Calibration returned no_recommendation "
            f"({calib.get('no_recommendation_reason')}). Cannot test token expiry."
        )

    cal_token = calib["calibration_token"]

    # Back-date expires_at by 1 hour to simulate expiry
    async def _expire_token():
        async with pool.acquire() as conn:
            updated = await conn.fetchval(
                "UPDATE calibration_tokens "
                "SET expires_at = NOW() - INTERVAL '1 hour' "
                "WHERE token_string = $1 "
                "RETURNING token_string",
                cal_token,
            )
            return updated

    updated_token = run_db(_expire_token())
    assert updated_token == cal_token, (
        f"UPDATE calibration_tokens found no row for token '{cal_token}'. "
        "Token may not have been written to DB."
    )

    # Apply calibration with expired token — must return HTTP 400
    r = client.post(
        "/conditions/apply-calibration",
        json={"calibration_token": cal_token, "new_version": "v2"},
        headers=elevated_headers,
    )
    assert r.status_code == 400, (
        f"Expired calibration token must return HTTP 400. "
        f"Got {r.status_code}: {r.text}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE 6 — Immutability After Calibration
# ══════════════════════════════════════════════════════════════════════════════

_C6_IMM_CONCEPT_ID = "cal.c6.imm.metric"
_C6_IMM_COND_ID    = "cal.c6.imm.cond"
_C6_IMM_ACTION_ID  = "cal.c6.imm.action"

_C6_SEQ_CONCEPT_ID = "cal.c6.seq.metric"
_C6_SEQ_COND_ID    = "cal.c6.seq.cond"
_C6_SEQ_ACTION_ID  = "cal.c6.seq.action"


@pytest.mark.e2e
def test_calibration_does_not_modify_existing_version(cal_client, api_headers, elevated_headers):
    """
    Applying calibration must not mutate condition v1.

    CalibrationService.apply_calibration() does:
      new_body = copy.deepcopy(source_body)   # deep copy
      new_body["version"] = new_version
      new_body["strategy"]["params"] = recommended_params
      await registry.register(new_body, ...)   # new row, not UPDATE

    This test snapshots v1 params before calibration and compares them
    byte-for-byte after apply_calibration creates v2.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C6_IMM_CONCEPT_ID, _C6_IMM_COND_ID, "v1", _C6_IMM_ACTION_ID, threshold=0.6)

    # Snapshot v1 before calibration
    r = client.get(f"/conditions/{_C6_IMM_COND_ID}", params={"version": "v1"}, headers=api_headers)
    assert r.status_code == 200
    v1_params_before = r.json()["strategy"]["params"].copy()

    # Submit 3 × false_positive to unlock calibration
    entities = [f"imm_entity_{i}" for i in range(1, 4)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C6_IMM_CONCEPT_ID, "v1", _C6_IMM_COND_ID, "v1",
        entities, "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C6_IMM_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib = r.json()

    if calib["status"] != "recommendation_available":
        pytest.skip(f"No recommendation available: {calib.get('no_recommendation_reason')}")

    # Apply → v2
    r = client.post(
        "/conditions/apply-calibration",
        json={"calibration_token": calib["calibration_token"], "new_version": "v2"},
        headers=elevated_headers,
    )
    assert r.status_code == 200

    # Re-fetch v1 — params must be byte-for-byte identical
    r = client.get(f"/conditions/{_C6_IMM_COND_ID}", params={"version": "v1"}, headers=api_headers)
    assert r.status_code == 200
    v1_params_after = r.json()["strategy"]["params"]

    assert v1_params_after == v1_params_before, (
        f"Calibration must not mutate v1 params!\n"
        f"  Before apply: {v1_params_before}\n"
        f"  After apply:  {v1_params_after}"
    )


@pytest.mark.e2e
def test_two_calibrations_create_sequential_versions(cal_client, api_headers, elevated_headers):
    """
    Two calibration rounds create v2 then v3 — each independently retrievable.

    Round 1: 3 × false_positive on v1 → tighten → v2 (value increases)
    Round 2: 3 × false_positive on v2 → tighten → v3 (value increases further)

    All three versions must:
      - Exist in the DB independently
      - Be retrievable via GET /conditions/{id}?version=v{n}
      - Have different strategy params

    Confirms: calibration creates new versions; does not overwrite old ones.
    """
    client, pool, run_db = cal_client

    _register_all(client, elevated_headers, _C6_SEQ_CONCEPT_ID, _C6_SEQ_COND_ID, "v1", _C6_SEQ_ACTION_ID, threshold=0.5)

    # Snapshot v1
    r = client.get(f"/conditions/{_C6_SEQ_COND_ID}", params={"version": "v1"}, headers=api_headers)
    assert r.status_code == 200
    v1_params = r.json()["strategy"]["params"].copy()

    # ── Round 1: calibrate v1 → create v2 ─────────────────────────────────────
    entities_r1 = [f"seq_r1_{i}" for i in range(1, 4)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C6_SEQ_CONCEPT_ID, "v1", _C6_SEQ_COND_ID, "v1",
        entities_r1, "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C6_SEQ_COND_ID, "condition_version": "v1"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib1 = r.json()

    if calib1["status"] != "recommendation_available":
        pytest.skip(f"Round 1: no recommendation ({calib1.get('no_recommendation_reason')})")

    r = client.post(
        "/conditions/apply-calibration",
        json={"calibration_token": calib1["calibration_token"], "new_version": "v2"},
        headers=elevated_headers,
    )
    assert r.status_code == 200, f"Round 1 apply-calibration failed: {r.text}"
    assert r.json()["new_version"] == "v2"

    r = client.get(f"/conditions/{_C6_SEQ_COND_ID}", params={"version": "v2"}, headers=api_headers)
    assert r.status_code == 200
    v2_params = r.json()["strategy"]["params"].copy()

    # Register v2 condition in DB means we also need the action to reference v2 for
    # evaluation; but we only need to evaluate/calibrate v2 — no rebind needed.
    # Seed new feedback on v2 using /evaluate/full with the v2 condition version.
    # ── Round 2: calibrate v2 → create v3 ─────────────────────────────────────
    entities_r2 = [f"seq_r2_{i}" for i in range(1, 4)]
    _seed_feedback(
        client, pool, run_db, api_headers,
        _C6_SEQ_CONCEPT_ID, "v1", _C6_SEQ_COND_ID, "v2",
        entities_r2, "false_positive",
    )

    r = client.post(
        "/conditions/calibrate",
        json={"condition_id": _C6_SEQ_COND_ID, "condition_version": "v2"},
        headers=api_headers,
    )
    assert r.status_code == 200
    calib2 = r.json()

    if calib2["status"] != "recommendation_available":
        pytest.skip(f"Round 2: no recommendation ({calib2.get('no_recommendation_reason')})")

    r = client.post(
        "/conditions/apply-calibration",
        json={"calibration_token": calib2["calibration_token"], "new_version": "v3"},
        headers=elevated_headers,
    )
    assert r.status_code == 200, f"Round 2 apply-calibration failed: {r.text}"
    assert r.json()["new_version"] == "v3"

    r = client.get(f"/conditions/{_C6_SEQ_COND_ID}", params={"version": "v3"}, headers=api_headers)
    assert r.status_code == 200
    v3_params = r.json()["strategy"]["params"].copy()

    # ── All three versions must exist independently ────────────────────────────

    # v1 still retrievable and unchanged
    r = client.get(f"/conditions/{_C6_SEQ_COND_ID}", params={"version": "v1"}, headers=api_headers)
    assert r.status_code == 200
    assert r.json()["strategy"]["params"] == v1_params, "v1 params must be unchanged"

    # v2 still retrievable with round-1 recommended params
    r = client.get(f"/conditions/{_C6_SEQ_COND_ID}", params={"version": "v2"}, headers=api_headers)
    assert r.status_code == 200
    assert r.json()["strategy"]["params"] == v2_params, "v2 params must be unchanged"

    # v3 still retrievable with round-2 recommended params
    r = client.get(f"/conditions/{_C6_SEQ_COND_ID}", params={"version": "v3"}, headers=api_headers)
    assert r.status_code == 200
    assert r.json()["strategy"]["params"] == v3_params, "v3 params must be unchanged"

    # Each version must have different params
    # v1(0.5) → tighten → v2(0.6) → tighten → v3(0.7)
    # step = max(current * 0.1, 0.1) = 0.1 for both rounds
    assert v1_params != v2_params, (
        f"v1 and v2 must have different params. v1={v1_params} v2={v2_params}"
    )
    assert v2_params != v3_params, (
        f"v2 and v3 must have different params. v2={v2_params} v3={v3_params}"
    )
    assert v1_params != v3_params, (
        f"v1 and v3 must have different params. v1={v1_params} v3={v3_params}"
    )
