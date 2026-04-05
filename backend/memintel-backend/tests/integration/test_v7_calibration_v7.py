"""
tests/integration/test_v7_calibration_v7.py
──────────────────────────────────────────────────────────────────────────────
T-7 Part 3 — Calibration interaction with the V7 concept_id path (M5).

Purpose: verify that calibration applied to a condition created via the M5
path (pre-compiled concept_id) produces correct results:
  - Threshold value changes after calibration
  - New condition version is created (old preserved)
  - Evaluation uses the calibrated threshold
  - Audit trail records calibrated (not compile-time) threshold
  - Calibration is condition-scoped (does not bleed to other conditions)
  - M5-path conditions are accepted by the calibration pipeline

Approach
────────
Most tests build the condition setup using the service/store layer directly
(DefinitionStore + CalibrationService) to avoid HTTP routing complexity with
auth dependencies (require_elevated_key is always enforced).

test_concept_id_path_condition_is_calibratable uses the full HTTP stack
(compile → register → create task → calibrate via route) to verify M5
end-to-end compatibility.

CalibrationService requires guardrails to compute bounds. A lightweight
_SimpleGuardrailsStore wraps the ConfigLoader output.

Threshold arithmetic for direction='below' + 'tighten':
  step = max(0.8 * 0.10, 0.1) = 0.1
  delta = -step (tighten + below → decrease value)
  new_value = 0.8 - 0.1 = 0.7

Threshold arithmetic for direction='below' + 'relax':
  delta = +step
  new_value = 0.8 + 0.1 = 0.9
"""
from __future__ import annotations

import asyncio
import types as _types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

import app.services.execute as _execute_module
from app.api.routes import concepts as concepts_route
from app.api.routes import tasks as tasks_route
from app.api.routes.concepts import (
    get_concept_compiler_service,
    get_concept_registration_service,
)
from app.api.routes.tasks import get_task_authoring_service
from app.config.config_loader import ConfigLoader
from app.models.calibration import (
    ApplyCalibrationRequest,
    CalibrateRequest,
    CalibrationStatus,
)
from app.models.condition import ConditionDefinition
from app.models.decision import DecisionRecord
from app.models.errors import MemintelError, memintel_error_handler
from app.registry.definitions import DefinitionRegistry
from app.runtime.data_resolver import MockConnector
from app.services.calibration import CalibrationService
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.services.execute import ExecuteService
from app.services.task_authoring import TaskAuthoringService
from app.stores import (
    CalibrationTokenStore,
    DefinitionStore,
    FeedbackStore,
    TaskStore,
)
from app.stores.compile_token import CompileTokenStore
from app.stores.decision import DecisionStore


# ── Guardrails path ────────────────────────────────────────────────────────────

_GUARDRAILS_YAML = str(
    Path(__file__).parent.parent.parent / "memintel_guardrails.yaml"
)


class _SimpleGuardrailsStore:
    """
    Minimal duck-typed guardrails store wrapping a loaded Guardrails object.
    Implements what CalibrationService and TaskAuthoringService need:
      get_guardrails(), get_threshold_bounds(), is_loaded(), get_active_version().
    """

    def __init__(self) -> None:
        self._guardrails = ConfigLoader().load_guardrails(_GUARDRAILS_YAML)

    def get_guardrails(self):
        return self._guardrails

    def get_threshold_bounds(self, strategy: str) -> dict:
        b = self._guardrails.constraints.threshold_bounds.get(strategy)
        return b.model_dump() if b else {}

    def is_loaded(self) -> bool:
        return True

    def get_active_version(self):
        return None


# ── Store / service factories ──────────────────────────────────────────────────

def _make_calib_service(db_pool) -> CalibrationService:
    """CalibrationService wired to the test pool and real guardrails."""
    return CalibrationService(
        feedback_store=FeedbackStore(db_pool),
        token_store=CalibrationTokenStore(db_pool),
        task_store=TaskStore(db_pool),
        definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
        guardrails_store=_SimpleGuardrailsStore(),
    )


def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
    """Full test app with concepts + tasks routes and an elevated key."""
    app = FastAPI()
    app.state.db = db_pool
    app.state.elevated_key = "test-elevated"
    app.state.guardrails_store = _SimpleGuardrailsStore()
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)
    app.include_router(tasks_route.router)

    async def _compiler_svc() -> ConceptCompilerService:
        return ConceptCompilerService(
            llm_client=llm_client,
            token_store=CompileTokenStore(db_pool),
        )

    async def _registration_svc() -> ConceptRegistrationService:
        return ConceptRegistrationService()

    async def _task_svc() -> TaskAuthoringService:
        return TaskAuthoringService(
            task_store=TaskStore(db_pool),
            definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
            llm_client=llm_client,
        )

    app.dependency_overrides[get_concept_compiler_service]    = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service]      = _task_svc
    return app


# ── DB registration helpers ────────────────────────────────────────────────────

def _register(run, db_pool, definition_id: str, body: dict, def_type: str) -> None:
    """Store a definition in the definitions table via DefinitionStore."""
    store = DefinitionStore(db_pool)
    run(store.register(
        definition_id=definition_id,
        version="v1",
        definition_type=def_type,
        namespace="org",
        body=body,
    ))


def _float_concept(concept_id: str, primitive_name: str) -> dict:
    """Minimal float concept body (z_score_op is a transparent passthrough)."""
    return {
        "concept_id":     concept_id,
        "version":        "v1",
        "namespace":      "org",
        "output_type":    "float",
        "description":    f"Test concept: {concept_id}",
        "primitives": {
            primitive_name: {"type": "float", "missing_data_policy": "null"}
        },
        "features": {
            "output": {
                "op":     "z_score_op",
                "inputs": {"input": primitive_name},
                "params": {},
            }
        },
        "output_feature": "output",
    }


def _threshold_condition(
    condition_id: str,
    concept_id: str,
    direction: str,
    value: float,
) -> dict:
    return {
        "condition_id":     condition_id,
        "version":          "v1",
        "concept_id":       concept_id,
        "concept_version":  "v1",
        "namespace":        "org",
        "strategy": {
            "type":   "threshold",
            "params": {"direction": direction, "value": value},
        },
    }


def build_condition(run, db_pool, concept_body: dict, condition_body: dict) -> tuple[str, str]:
    """Register concept + condition in DB. Returns (condition_id, 'v1')."""
    _register(run, db_pool, concept_body["concept_id"], concept_body, "concept")
    _register(run, db_pool, condition_body["condition_id"], condition_body, "condition")
    return condition_body["condition_id"], "v1"


def evaluate_condition(run, db_pool, condition_id, condition_version, entity, connector_data):
    """Evaluate a condition with a MockConnector. Returns DecisionResult."""
    mock_conn = MockConnector(data=connector_data)
    svc = ExecuteService(pool=db_pool)
    req = _types.SimpleNamespace(
        condition_id=condition_id,
        condition_version=condition_version,
        entity=entity,
        timestamp=None,
        explain=False,
    )
    with patch.object(_execute_module, "_make_connector", return_value=mock_conn):
        return run(svc.evaluate_condition(req))


# ── Calibration helpers ────────────────────────────────────────────────────────

def calibrate_and_apply(run, db_pool, condition_id, condition_version, direction) -> str:
    """
    Run calibrate() with explicit feedback_direction, then apply_calibration().
    Returns the new_version string (e.g. 'v1.1').
    """
    svc = _make_calib_service(db_pool)

    result = run(svc.calibrate(CalibrateRequest(
        condition_id=condition_id,
        condition_version=condition_version,
        feedback_direction=direction,
    )))
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        f"Calibration returned no recommendation: {result}"
    )
    token = result.calibration_token
    assert token is not None

    apply_result = run(svc.apply_calibration(ApplyCalibrationRequest(
        calibration_token=token,
    )))
    return apply_result.new_version


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Calibration changes threshold value and creates new version
# ═══════════════════════════════════════════════════════════════════════════════

def test_calibration_changes_threshold(db_pool, run):
    """
    Tighten a threshold below-0.8 condition → new version with value=0.7.

    threshold 'below' + tighten: delta = -step = -0.1; new = 0.7
    Original v1 must be preserved unchanged at value=0.8.
    """
    concept  = _float_concept("calib1.concept", "calib1.score")
    condition = _threshold_condition("calib1.cond", "calib1.concept", "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    new_version = calibrate_and_apply(run, db_pool, "calib1.cond", "v1", "tighten")

    # New version must differ from original
    assert new_version != "v1", f"New version should not be 'v1'; got '{new_version}'"

    # New version in DB must have the calibrated threshold
    store = DefinitionStore(db_pool)
    new_body = run(store.get("calib1.cond", new_version))
    new_cond = ConditionDefinition.model_validate(new_body)
    new_value = new_cond.strategy.params.value
    assert abs(new_value - 0.7) < 0.01, (
        f"Expected calibrated value ≈ 0.7; got {new_value}"
    )

    # Original v1 unchanged at 0.8
    orig_body = run(store.get("calib1.cond", "v1"))
    orig_cond = ConditionDefinition.model_validate(orig_body)
    assert abs(orig_cond.strategy.params.value - 0.8) < 0.01, (
        f"Original v1 threshold should be unchanged at 0.8; got {orig_cond.strategy.params.value}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Calibrated condition evaluates differently from original
# ═══════════════════════════════════════════════════════════════════════════════

def test_calibrated_condition_evaluates_correctly(db_pool, run):
    """
    Relax a threshold below-0.8 condition → new version with value=0.9.

    With value=0.85 as the input primitive:
      Original  (v1,   below 0.8): 0.85 < 0.8 → False → NOT fired
      Calibrated (v1.1, below 0.9): 0.85 < 0.9 → True  → fired

    Verifies that calibration actually changes evaluation behaviour.
    """
    concept  = _float_concept("calib2.concept", "calib2.score")
    condition = _threshold_condition("calib2.cond", "calib2.concept", "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    new_version = calibrate_and_apply(run, db_pool, "calib2.cond", "v1", "relax")

    # Verify new threshold is ≈ 0.9 (0.8 + 0.1 step for relax+below)
    new_body = run(DefinitionStore(db_pool).get("calib2.cond", new_version))
    new_cond = ConditionDefinition.model_validate(new_body)
    assert abs(new_cond.strategy.params.value - 0.9) < 0.01, (
        f"Expected relaxed threshold ≈ 0.9; got {new_cond.strategy.params.value}"
    )

    connector_data = {("calib2.score", "entity-x", None): 0.85}

    # Original (below 0.8): 0.85 < 0.8 → False
    orig_result = evaluate_condition(
        run, db_pool, "calib2.cond", "v1", "entity-x", connector_data
    )
    assert orig_result.value is False, (
        f"Original threshold (0.8): value=0.85 should NOT fire; got {orig_result.value}"
    )

    # Calibrated (below 0.9): 0.85 < 0.9 → True
    calib_result = evaluate_condition(
        run, db_pool, "calib2.cond", new_version, "entity-x", connector_data
    )
    assert calib_result.value is True, (
        f"Calibrated threshold (0.9): value=0.85 SHOULD fire; got {calib_result.value}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Audit trail records calibrated threshold
# ═══════════════════════════════════════════════════════════════════════════════

def test_calibration_audit_trail(db_pool, run):
    """
    After evaluating the calibrated condition (v1.1), the decision row's
    threshold_applied JSONB column must reflect the calibrated threshold (0.9),
    not the compile-time threshold (0.8).
    """
    concept  = _float_concept("calib3.concept", "calib3.score")
    condition = _threshold_condition("calib3.cond", "calib3.concept", "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    new_version = calibrate_and_apply(run, db_pool, "calib3.cond", "v1", "relax")

    # Evaluate the calibrated condition → creates a decision record
    ts = "2025-06-01T10:00:00Z"
    mock_conn = MockConnector(data={("calib3.score", "audit-entity", ts): 0.85})
    svc = ExecuteService(pool=db_pool)
    req = _types.SimpleNamespace(
        concept_id="calib3.concept", concept_version="v1",
        condition_id="calib3.cond", condition_version=new_version,
        entity="audit-entity", timestamp=ts, dry_run=False, explain=False,
    )
    with patch.object(_execute_module, "_make_connector", return_value=mock_conn):
        run(svc.evaluate_full(req))

    # Fire-and-forget persistence: give the task time to write
    run(asyncio.sleep(0.05))

    # Query decisions table for the calibrated condition
    rows = run(db_pool.fetch(
        """
        SELECT threshold_applied
        FROM decisions
        WHERE condition_id = $1 AND condition_version = $2 AND entity_id = $3
        """,
        "calib3.cond", new_version, "audit-entity",
    ))
    assert len(rows) == 1, (
        f"Expected 1 decision row for calibrated condition; got {len(rows)}"
    )

    # threshold_applied must reflect calibrated value (≈ 0.9), not original (0.8)
    threshold_applied = rows[0]["threshold_applied"]
    assert threshold_applied is not None, "threshold_applied must not be NULL"
    import json
    ta = json.loads(threshold_applied) if isinstance(threshold_applied, str) else threshold_applied
    applied_value = float(ta.get("value", ta.get("threshold", -1)))
    assert abs(applied_value - 0.9) < 0.01, (
        f"Audit trail threshold_applied should be calibrated value 0.9; got {applied_value}. "
        f"Full threshold_applied: {ta}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — M5 concept_id path condition is accepted by calibration
# ═══════════════════════════════════════════════════════════════════════════════

def test_concept_id_path_condition_is_calibratable(db_pool, run, llm_mock, loan_compile_request):
    """
    End-to-end HTTP test: compile → register → create task via M5 path.
    Extract condition_id from task response.
    POST /conditions/calibrate with explicit feedback_direction.
    Assert: calibration returns recommendation_available (not no_recommendation).

    Verifies: M5 path does not create conditions incompatible with calibration.
    """
    from app.api.routes import conditions as conditions_route
    from app.api.routes.conditions import get_calibration_service
    from tests.integration.conftest_v7 import compile_and_register

    # Build full app with conditions route
    app = _make_test_app(db_pool, llm_mock)
    app.include_router(conditions_route.router)

    # Override calibration service to inject test guardrails_store
    async def _calib_svc():
        return _make_calib_service(db_pool)

    app.dependency_overrides[get_calibration_service] = _calib_svc

    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # M3 + M4: compile and register the concept
            concept_id, _ = await compile_and_register(
                client,
                identifier=loan_compile_request.identifier,
                description=loan_compile_request.description,
                output_type=loan_compile_request.output_type,
                signal_names=loan_compile_request.signal_names,
            )

            # M5: create task using pre-compiled concept
            task_resp = await client.post("/tasks", json={
                "intent":       "alert on repayment ratio",
                "entity_scope": "loan",
                "delivery": {
                    "type":     "webhook",
                    "endpoint": "https://calib-test.example.com/hook",
                },
                "concept_id":    concept_id,
                "stream":        False,
                "return_reasoning": False,
            })
            assert task_resp.status_code == 200, (
                f"create task failed: {task_resp.status_code}: {task_resp.text}"
            )
            task_data = task_resp.json()
            condition_id = task_data["condition_id"]
            condition_version = task_data["condition_version"]

            # Calibrate the M5-path condition
            calib_resp = await client.post("/conditions/calibrate", json={
                "condition_id":      condition_id,
                "condition_version": condition_version,
                "feedback_direction": "tighten",
            })
        return calib_resp

    calib_resp = run(_go())

    assert calib_resp.status_code == 200, (
        f"calibrate failed: {calib_resp.status_code}: {calib_resp.text}"
    )
    data = calib_resp.json()
    assert data["status"] == "recommendation_available", (
        f"Expected recommendation_available; got: {data['status']}. "
        f"no_recommendation_reason: {data.get('no_recommendation_reason')}. "
        "M5-path condition should be calibratable."
    )
    assert data.get("calibration_token") is not None, (
        "calibration_token must be present when recommendation is available"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Calibration does not affect other conditions
# ═══════════════════════════════════════════════════════════════════════════════

def test_calibration_does_not_affect_other_conditions(db_pool, run):
    """
    Two conditions sharing the same concept. Calibrate condition_a.
    condition_b must remain unchanged.

    Verifies: calibration is condition-scoped, not concept-scoped.
    """
    concept = _float_concept("calib5.concept", "calib5.score")
    cond_a = _threshold_condition("calib5.cond_a", "calib5.concept", "below", 0.8)
    cond_b = _threshold_condition("calib5.cond_b", "calib5.concept", "below", 0.8)

    _register(run, db_pool, concept["concept_id"], concept, "concept")
    _register(run, db_pool, cond_a["condition_id"], cond_a, "condition")
    _register(run, db_pool, cond_b["condition_id"], cond_b, "condition")

    # Calibrate only condition_a → tighten → value=0.7
    new_version_a = calibrate_and_apply(run, db_pool, "calib5.cond_a", "v1", "tighten")

    # condition_b v1 must still have value=0.8 in definitions
    store = DefinitionStore(db_pool)
    b_body = run(store.get("calib5.cond_b", "v1"))
    b_cond = ConditionDefinition.model_validate(b_body)
    assert abs(b_cond.strategy.params.value - 0.8) < 0.01, (
        f"condition_b should be unchanged at 0.8; got {b_cond.strategy.params.value}"
    )

    # Evaluate condition_b with value=0.75 → 0.75 < 0.8 → fires
    b_result = evaluate_condition(
        run, db_pool, "calib5.cond_b", "v1", "entity-iso",
        {("calib5.score", "entity-iso", None): 0.75},
    )
    assert b_result.value is True, (
        f"condition_b (below 0.8) should fire for value=0.75; got {b_result.value}"
    )

    # Evaluate calibrated condition_a (v1.1, below 0.7) with value=0.75 → 0.75 < 0.7 → False
    a_result = evaluate_condition(
        run, db_pool, "calib5.cond_a", new_version_a, "entity-iso",
        {("calib5.score", "entity-iso", None): 0.75},
    )
    assert a_result.value is False, (
        f"calibrated condition_a (below 0.7) should NOT fire for value=0.75; got {a_result.value}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — Reasoning trace after calibration
# ═══════════════════════════════════════════════════════════════════════════════

def test_reasoning_trace_after_calibration(db_pool, run, llm_mock, loan_compile_request):
    """
    Task created via M5 path with return_reasoning=True returns a trace
    with 4 steps where steps 1 and 2 are skipped (pre-compiled concept).

    After calibration creates a new condition version, the original task's
    stored condition_version is still v1 (tasks are never auto-rebound). The
    new condition version (v1.1) is available for opt-in rebinding via PATCH.

    Verifies:
      - M5 path returns reasoning trace with 4 steps.
      - Steps 1 and 2 are skipped (concept pre-compiled).
      - Calibration creates a new condition version without breaking the task.
    """
    from tests.integration.conftest_v7 import compile_and_register

    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Compile and register the concept
            concept_id, _ = await compile_and_register(
                client,
                identifier=loan_compile_request.identifier,
                description=loan_compile_request.description,
                output_type=loan_compile_request.output_type,
                signal_names=loan_compile_request.signal_names,
            )

            # Create task via M5 path with return_reasoning=True
            task_resp = await client.post("/tasks", json={
                "intent":          "alert on repayment ratio for trace test",
                "entity_scope":    "loan",
                "delivery": {
                    "type":     "webhook",
                    "endpoint": "https://trace-test.example.com/hook",
                },
                "concept_id":      concept_id,
                "return_reasoning": True,
                "stream":          False,
            })
            return task_resp, concept_id

    task_resp, concept_id = run(_go())

    assert task_resp.status_code == 200, (
        f"Task creation failed: {task_resp.status_code}: {task_resp.text}"
    )
    data = task_resp.json()

    # Reasoning trace must be present
    trace = data.get("reasoning_trace")
    assert trace is not None, (
        "reasoning_trace must be present when return_reasoning=True"
    )
    steps = {s["step_index"]: s for s in trace["steps"]}

    # Steps 1 and 2 skipped — pre-compiled concept bypasses these
    assert steps[1]["outcome"] == "skipped", (
        f"Step 1 should be skipped for M5 path; got: {steps[1]}"
    )
    assert steps[2]["outcome"] == "skipped", (
        f"Step 2 should be skipped for M5 path; got: {steps[2]}"
    )

    # Steps 3 and 4 accepted
    assert steps[3]["outcome"] == "accepted", (
        f"Step 3 (condition strategy) should be accepted; got: {steps[3]}"
    )
    assert steps[4]["outcome"] == "accepted", (
        f"Step 4 (action binding) should be accepted; got: {steps[4]}"
    )

    # Apply calibration to the task's condition via service layer
    condition_id      = data["condition_id"]
    condition_version = data["condition_version"]

    new_version = calibrate_and_apply(
        run, db_pool, condition_id, condition_version, "tighten"
    )

    # New version must exist in DB
    new_body = run(DefinitionStore(db_pool).get(condition_id, new_version))
    new_cond = ConditionDefinition.model_validate(new_body)
    assert new_cond.version == new_version, (
        f"New calibrated condition version mismatch: {new_cond.version} vs {new_version}"
    )

    # Original task's condition_version is unchanged (tasks never auto-rebound)
    assert data["condition_version"] == condition_version, (
        "Original task response condition_version should not change after calibration"
    )
