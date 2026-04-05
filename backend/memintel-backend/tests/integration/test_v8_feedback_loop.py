"""
tests/integration/test_v8_feedback_loop.py
──────────────────────────────────────────────────────────────────────────────
T-8 Part 2 — Feedback loop integrity.

Covers the three-step calibration chain (feedback → calibrate → apply) under
edge conditions and failure scenarios.  All tests use:
  - Real asyncpg pool (no DB mocks)
  - Direct service layer for calibrate / apply (bypasses require_elevated_key)
  - HTTP via httpx.AsyncClient for feedback and task-creation routes

─────────────────────────────────────────────────────────────────────────────
Q1–Q6 answers (from source review required by the spec)
─────────────────────────────────────────────────────────────────────────────

Q1. What does POST /feedback/decision store, and what must exist first?
    Stores a FeedbackRecord in the feedback_records table (column:
    decision_timestamp ↔ Python field: timestamp).  Two things must exist first:
      a) The condition (id, version) must be registered in definitions.
      b) A DecisionRecord must exist in decisions for
         (condition_id, condition_version, entity_id, evaluated_at=timestamp).
    FeedbackService raises NotFoundError → HTTP 404 if either is absent.

Q2. When does POST /conditions/calibrate return recommendation vs no_recommendation?
    Always returns HTTP 200.  Returns no_recommendation when:
      - Strategy is 'equals' or 'composite'       → not_applicable_strategy
      - Fewer than MIN_FEEDBACK_THRESHOLD=3 records AND no feedback_direction
        override                                  → insufficient_data
      - Adjustment would exceed guardrail bounds AND on_bounds_exceeded='reject'
                                                  → bounds_exceeded
      - guardrails_store is None                  → guardrails_unavailable
    Returns recommendation_available when direction resolves (explicit override
    or majority vote on ≥3 records) and guardrails are loaded.

Q3. Does POST /conditions/apply-calibration mutate the existing condition?
    No.  It creates a NEW immutable condition version (deep copy with new params).
    The source version is preserved unchanged.  Auto-increments the last numeric
    component of the version string: 'v1'→'v1.1', '1.0'→'1.1', '2'→'3'.

Q4. Is apply-calibration idempotent?
    No.  The calibration token is consumed atomically on the first call by
    CalibrationTokenStore.resolve_and_invalidate().  A second call with the same
    token returns None → MemintelError(PARAMETER_ERROR) → HTTP 400.

Q5. What happens when calibrate() is called with zero feedback records and no
    explicit feedback_direction?
    derive_direction([]) → len([]) < MIN_FEEDBACK_THRESHOLD=3 → returns None →
    CalibrationResult(status=no_recommendation, reason=insufficient_data).
    HTTP 200 (always).

Q6. What does apply-calibration return when the token does not exist?
    resolve_and_invalidate(unknown_token) returns None → the service raises
    MemintelError(PARAMETER_ERROR, "Invalid or expired calibration token.") →
    HTTP 400.
"""
from __future__ import annotations

import types as _types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

import app.services.execute as _execute_module
from app.api.routes import concepts as concepts_route
from app.api.routes import feedback as feedback_route
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
    FeedbackRecord,
    FeedbackRequest,
    FeedbackValue,
    MIN_FEEDBACK_THRESHOLD,
    NoRecommendationReason,
)
from app.models.condition import ConditionDefinition
from app.models.decision import DecisionRecord
from app.models.errors import (
    ConflictError,
    MemintelError,
    NotFoundError,
    memintel_error_handler,
)
from app.registry.definitions import DefinitionRegistry
from app.runtime.data_resolver import MockConnector
from app.services.calibration import CalibrationService
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.services.execute import ExecuteService
from app.services.feedback import FeedbackService
from app.services.task_authoring import TaskAuthoringService
from app.stores import (
    CalibrationTokenStore,
    ContextStore,
    DefinitionStore,
    FeedbackStore,
    TaskStore,
)
from app.stores.compile_token import CompileTokenStore
from app.stores.decision import DecisionStore

from tests.integration.conftest_v7 import LLMMockClient

# ── Guardrails path ─────────────────────────────────────────────────────────────

_GUARDRAILS_YAML = str(
    Path(__file__).parent.parent.parent / "memintel_guardrails.yaml"
)


class _SimpleGuardrailsStore:
    """Minimal duck-typed guardrails store for tests."""

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


# ── Known test timestamps (ISO 8601 UTC) ────────────────────────────────────────
# Fixed timestamps allow exact decision lookup after direct DB insertion.

_TS = [
    "2025-01-15T12:00:00+00:00",
    "2025-01-15T12:01:00+00:00",
    "2025-01-15T12:02:00+00:00",
    "2025-01-15T12:03:00+00:00",
    "2025-01-15T12:04:00+00:00",
    "2025-01-15T12:05:00+00:00",
]


# ── Service factories ────────────────────────────────────────────────────────────

def _make_calib_service(db_pool) -> CalibrationService:
    """CalibrationService wired to the test pool and real guardrails."""
    return CalibrationService(
        feedback_store=FeedbackStore(db_pool),
        token_store=CalibrationTokenStore(db_pool),
        task_store=TaskStore(db_pool),
        definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
        guardrails_store=_SimpleGuardrailsStore(),
    )


def _make_feedback_service(db_pool) -> FeedbackService:
    """FeedbackService wired to the test pool."""
    return FeedbackService(
        feedback_store=FeedbackStore(db_pool),
        definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
        decision_store=DecisionStore(db_pool),
    )


# ── App factories ────────────────────────────────────────────────────────────────

def _make_feedback_app(db_pool) -> FastAPI:
    """Minimal FastAPI app with feedback route for HTTP validation tests."""
    app = FastAPI()
    app.state.db = db_pool
    # api_key not set → require_api_key passes all requests (dev mode)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(feedback_route.router)
    return app


def _make_full_app(db_pool, llm_client: Any) -> FastAPI:
    """Full app: concepts + tasks + feedback for V7 interaction tests."""
    app = FastAPI()
    app.state.db = db_pool
    app.state.elevated_key = "test-elevated"
    app.state.guardrails_store = _SimpleGuardrailsStore()
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)
    app.include_router(tasks_route.router)
    app.include_router(feedback_route.router)

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

    app.dependency_overrides[get_concept_compiler_service]     = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service]       = _task_svc
    return app


# ── DB helpers ───────────────────────────────────────────────────────────────────

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
        "condition_id":    condition_id,
        "version":         "v1",
        "concept_id":      concept_id,
        "concept_version": "v1",
        "namespace":       "org",
        "strategy": {
            "type":   "threshold",
            "params": {"direction": direction, "value": value},
        },
    }


def _register(
    run,
    db_pool,
    definition_id: str,
    body: dict,
    def_type: str,
    version: str = "v1",
) -> None:
    """Store a definition in the definitions table."""
    store = DefinitionStore(db_pool)
    run(store.register(
        definition_id=definition_id,
        version=version,
        definition_type=def_type,
        namespace="org",
        body=body,
    ))


def build_condition(run, db_pool, concept_body: dict, condition_body: dict) -> tuple[str, str]:
    """Register concept + condition.  Returns (condition_id, 'v1')."""
    _register(run, db_pool, concept_body["concept_id"], concept_body, "concept")
    _register(run, db_pool, condition_body["condition_id"], condition_body, "condition")
    return condition_body["condition_id"], "v1"


def insert_decision(
    run,
    db_pool,
    condition_id: str,
    condition_version: str,
    concept_id: str,
    entity: str,
    fired: bool,
    timestamp_iso: str,
) -> str:
    """
    Insert a decision record with a known evaluated_at timestamp.

    The timestamp is set explicitly so feedback can reference it exactly.
    Returns the decision_id (or '' if ON CONFLICT DO NOTHING fired).
    """
    store = DecisionStore(db_pool)
    record = DecisionRecord(
        concept_id=concept_id,
        concept_version="v1",
        condition_id=condition_id,
        condition_version=condition_version,
        entity_id=entity,
        evaluated_at=datetime.fromisoformat(timestamp_iso),
        fired=fired,
        dry_run=False,
    )
    return run(store.record(record))


def submit_feedback(
    run,
    db_pool,
    condition_id: str,
    condition_version: str,
    entity: str,
    timestamp_iso: str,
    feedback_value: str,
) -> Any:
    """Submit feedback via FeedbackService (service layer, bypasses HTTP)."""
    svc = _make_feedback_service(db_pool)
    req = FeedbackRequest(
        condition_id=condition_id,
        condition_version=condition_version,
        entity=entity,
        timestamp=timestamp_iso,
        feedback=FeedbackValue(feedback_value),
    )
    return run(svc.submit(req))


def calibrate(run, db_pool, condition_id: str, version: str, **kwargs) -> Any:
    """Run CalibrationService.calibrate()."""
    svc = _make_calib_service(db_pool)
    req = CalibrateRequest(
        condition_id=condition_id,
        condition_version=version,
        **kwargs,
    )
    return run(svc.calibrate(req))


def apply_calibration(run, db_pool, token: str, new_version: str | None = None) -> Any:
    """Run CalibrationService.apply_calibration()."""
    svc = _make_calib_service(db_pool)
    req = ApplyCalibrationRequest(
        calibration_token=token,
        **({"new_version": new_version} if new_version else {}),
    )
    return run(svc.apply_calibration(req))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Full three-step calibration chain (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════════


def test_full_calibration_chain_tighten(db_pool, run):
    """
    3 false_positive feedbacks → calibrate → apply → threshold tightened.

    Threshold arithmetic (direction='below', tighten):
      step  = max(0.8 × 0.10, 0.1) = 0.1
      delta = -step  (below + tighten → decrease threshold)
      new   = 0.8 − 0.1 = 0.7

    Verifies:
      - All 3 feedback records accepted (status='recorded')
      - calibrate() returns recommendation_available + non-None token
      - apply_calibration() returns a new version
      - New version has threshold ≈ 0.7
      - Original v1 is preserved at 0.8
    """
    cid = "chain1.concept"
    eid = "chain1.condition"
    concept   = _float_concept(cid, "chain1.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    entity = "entity-fp-001"
    for ts in _TS[:3]:
        insert_decision(run, db_pool, eid, "v1", cid, entity, fired=True, timestamp_iso=ts)
        resp = submit_feedback(run, db_pool, eid, "v1", entity, ts, "false_positive")
        assert resp.status == "recorded", f"Expected 'recorded'; got '{resp.status}' for ts={ts}"

    # Step 2: Calibrate
    result = calibrate(run, db_pool, eid, "v1")
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        f"Expected recommendation_available; got {result.status}, reason={result.no_recommendation_reason}"
    )
    assert result.calibration_token is not None, "calibration_token must not be None"
    assert result.recommended_params is not None, "recommended_params must not be None"

    # Step 3: Apply
    apply_result = apply_calibration(run, db_pool, result.calibration_token)
    new_version = apply_result.new_version
    assert new_version != "v1", f"new_version must differ from 'v1'; got '{new_version}'"
    assert apply_result.previous_version == "v1"

    # Verify DB state
    store = DefinitionStore(db_pool)
    new_body = run(store.get(eid, new_version))
    assert new_body is not None, f"New version '{new_version}' not found in definitions table"
    new_val = new_body["strategy"]["params"]["value"]
    assert abs(new_val - 0.7) < 0.01, (
        f"Expected tightened threshold ≈ 0.7; got {new_val}"
    )

    # Original v1 must be unchanged at 0.8
    orig_body = run(store.get(eid, "v1"))
    orig_val  = orig_body["strategy"]["params"]["value"]
    assert abs(orig_val - 0.8) < 0.01, (
        f"Original v1 threshold must be unchanged at 0.8; got {orig_val}"
    )


def test_full_calibration_chain_relax(db_pool, run):
    """
    3 false_negative feedbacks → calibrate → apply → threshold relaxed.

    Threshold arithmetic (direction='below', relax):
      step  = 0.1
      delta = +step
      new   = 0.8 + 0.1 = 0.9
    """
    cid = "chain2.concept"
    eid = "chain2.condition"
    concept   = _float_concept(cid, "chain2.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    entity = "entity-fn-001"
    for ts in _TS[:3]:
        insert_decision(run, db_pool, eid, "v1", cid, entity, fired=False, timestamp_iso=ts)
        resp = submit_feedback(run, db_pool, eid, "v1", entity, ts, "false_negative")
        assert resp.status == "recorded"

    result = calibrate(run, db_pool, eid, "v1")
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        f"Expected recommendation; got {result.status}"
    )

    apply_result = apply_calibration(run, db_pool, result.calibration_token)
    new_version = apply_result.new_version
    assert new_version != "v1"

    new_body = run(DefinitionStore(db_pool).get(eid, new_version))
    new_val = new_body["strategy"]["params"]["value"]
    assert abs(new_val - 0.9) < 0.01, (
        f"Expected relaxed threshold ≈ 0.9; got {new_val}"
    )


def test_calibration_chain_explicit_direction_bypass(db_pool, run):
    """
    calibrate() with explicit feedback_direction bypasses feedback aggregation.

    No decisions or feedback records are inserted.  feedback_direction='tighten'
    is passed directly → immediate recommendation_available without stored feedback.
    This verifies the override path in CalibrationService.calibrate().
    """
    cid = "chain3.concept"
    eid = "chain3.condition"
    concept   = _float_concept(cid, "chain3.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    # No feedback records — override with explicit direction
    result = calibrate(run, db_pool, eid, "v1", feedback_direction="tighten")
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        "With explicit feedback_direction, recommendation must always be available "
        f"(got {result.status}, reason={result.no_recommendation_reason})"
    )
    token = result.calibration_token
    assert token is not None

    apply_result = apply_calibration(run, db_pool, token)
    new_version  = apply_result.new_version
    assert new_version != "v1"

    # Verify tightened threshold in DB
    new_body = run(DefinitionStore(db_pool).get(eid, new_version))
    new_val  = new_body["strategy"]["params"]["value"]
    assert abs(new_val - 0.7) < 0.01, (
        f"Expected tightened threshold ≈ 0.7 after explicit override; got {new_val}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Feedback edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_feedback_nonexistent_condition(db_pool, run):
    """
    Feedback for unregistered condition → NotFoundError (HTTP 404).

    FeedbackService.submit() checks registry.get() BEFORE any decision lookup.
    The condition 'phantom.condition' is never registered — expect NotFoundError.
    """
    svc = _make_feedback_service(db_pool)
    req = FeedbackRequest(
        condition_id="phantom.condition",
        condition_version="v1",
        entity="entity-x",
        timestamp=_TS[0],
        feedback=FeedbackValue.FALSE_POSITIVE,
    )
    with pytest.raises(NotFoundError):
        run(svc.submit(req))


def test_feedback_decision_not_found(db_pool, run):
    """
    Feedback for registered condition but no matching decision → NotFoundError.

    The condition is registered, but no decision exists in the decisions table
    for (condition_id, condition_version, entity, timestamp).
    """
    cid = "feedback2.concept"
    eid = "feedback2.condition"
    concept   = _float_concept(cid, "feedback2.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    # No decision record inserted — feedback must fail
    svc = _make_feedback_service(db_pool)
    req = FeedbackRequest(
        condition_id=eid,
        condition_version="v1",
        entity="entity-no-decision",
        timestamp=_TS[0],
        feedback=FeedbackValue.FALSE_POSITIVE,
    )
    with pytest.raises(NotFoundError) as exc_info:
        run(svc.submit(req))
    assert "No decision record found" in str(exc_info.value), (
        f"Expected 'No decision record found' in message; got: {exc_info.value}"
    )


def test_feedback_duplicate_rejected(db_pool, run):
    """
    Duplicate feedback for the same decision → ConflictError (HTTP 409).

    Uniqueness key: (condition_id, condition_version, entity, timestamp).
    The first submission succeeds; the second raises ConflictError.
    """
    cid = "feedback3.concept"
    eid = "feedback3.condition"
    concept   = _float_concept(cid, "feedback3.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    entity = "entity-dup"
    insert_decision(run, db_pool, eid, "v1", cid, entity, fired=True, timestamp_iso=_TS[0])

    svc = _make_feedback_service(db_pool)
    req = FeedbackRequest(
        condition_id=eid,
        condition_version="v1",
        entity=entity,
        timestamp=_TS[0],
        feedback=FeedbackValue.FALSE_POSITIVE,
    )

    # First submission: accepted
    result1 = run(svc.submit(req))
    assert result1.status == "recorded"
    assert result1.feedback_id, "Expected non-empty feedback_id on first submission"

    # Second submission: rejected
    with pytest.raises(ConflictError) as exc_info:
        run(svc.submit(req))
    assert "Feedback already submitted" in str(exc_info.value), (
        f"Expected 'Feedback already submitted'; got: {exc_info.value}"
    )


def test_feedback_invalid_value_via_http(db_pool, run):
    """
    POST /feedback/decision with feedback='useful' → HTTP 422 (Pydantic validation).

    Pydantic rejects 'useful' at the HTTP layer (not a valid FeedbackValue)
    before the route handler is called.  FastAPI returns 422 Unprocessable Entity.
    """
    app       = _make_feedback_app(db_pool)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/feedback/decision", json={
                "condition_id":      "any.condition",
                "condition_version": "v1",
                "entity":            "entity-x",
                "timestamp":         _TS[0],
                "feedback":          "useful",   # not a valid FeedbackValue
            })

    resp = run(_go())
    assert resp.status_code == 422, (
        f"Expected HTTP 422 for invalid feedback value; got {resp.status_code}: {resp.text}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Calibrate edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_calibrate_zero_feedback_no_recommendation(db_pool, run):
    """
    Calibrate with zero feedback records → no_recommendation(insufficient_data).

    derive_direction([]) → len([]) < MIN_FEEDBACK_THRESHOLD=3 → returns None.
    Always returns HTTP 200 — inspect the status field.
    """
    cid = "calib_zero.concept"
    eid = "calib_zero.condition"
    concept   = _float_concept(cid, "calib_zero.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    result = calibrate(run, db_pool, eid, "v1")
    assert result.status == CalibrationStatus.NO_RECOMMENDATION, (
        f"Expected no_recommendation; got {result.status}"
    )
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA, (
        f"Expected insufficient_data; got {result.no_recommendation_reason}"
    )
    assert result.calibration_token is None, "No token expected on no_recommendation"


def test_calibrate_two_feedback_no_recommendation(db_pool, run):
    """
    Calibrate with 2 feedback records (< MIN_FEEDBACK_THRESHOLD=3) → no_recommendation.

    2 false_positive records are inserted — still below the threshold.
    derive_direction([r1, r2]) → len == 2 < 3 → returns None → insufficient_data.
    """
    cid = "calib_two.concept"
    eid = "calib_two.condition"
    concept   = _float_concept(cid, "calib_two.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    entity = "entity-two"
    for ts in _TS[:2]:     # only 2 records — intentionally below threshold
        insert_decision(run, db_pool, eid, "v1", cid, entity, fired=True, timestamp_iso=ts)
        submit_feedback(run, db_pool, eid, "v1", entity, ts, "false_positive")

    result = calibrate(run, db_pool, eid, "v1")
    assert result.status == CalibrationStatus.NO_RECOMMENDATION, (
        f"Expected no_recommendation with 2 records; got {result.status}"
    )
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA, (
        f"Expected insufficient_data; got {result.no_recommendation_reason}"
    )


def test_calibrate_preview_not_applied(db_pool, run):
    """
    calibrate() returns a token but does NOT create a new version in the DB.

    The token is a preview — apply_calibration() must be called separately to
    create the new condition version.  After calibrate(), only 'v1' exists.
    """
    cid = "calib_preview.concept"
    eid = "calib_preview.condition"
    concept   = _float_concept(cid, "calib_preview.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    result = calibrate(run, db_pool, eid, "v1", feedback_direction="tighten")
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.calibration_token is not None

    # Auto-incremented would be 'v1.1' — must NOT exist yet
    store = DefinitionStore(db_pool)
    preview_body = run(store.get(eid, "v1.1"))
    assert preview_body is None, (
        "calibrate() must NOT create a new condition version — "
        f"found unexpected body for v1.1: {preview_body}"
    )

    # 'v1' still exists and is unchanged
    orig_body = run(store.get(eid, "v1"))
    assert orig_body is not None, "Original v1 must still exist after calibrate()"
    assert abs(orig_body["strategy"]["params"]["value"] - 0.8) < 0.01


def test_calibrate_nonexistent_condition(db_pool, run):
    """
    Calibrate on unregistered condition_id → NotFoundError (HTTP 404).

    CalibrationService.calibrate() calls registry.get() first — raises
    NotFoundError if the condition is not registered.
    """
    svc = _make_calib_service(db_pool)
    req = CalibrateRequest(
        condition_id="ghost.condition",
        condition_version="v1",
    )
    with pytest.raises(NotFoundError):
        run(svc.calibrate(req))


def test_calibrate_token_single_use(db_pool, run):
    """
    apply_calibration() with an already-used token → MemintelError (HTTP 400).

    The first apply consumes the token atomically.  A second apply with the
    same token raises PARAMETER_ERROR ("Invalid or expired calibration token").
    """
    cid = "single_use.concept"
    eid = "single_use.condition"
    concept   = _float_concept(cid, "single_use.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    result = calibrate(run, db_pool, eid, "v1", feedback_direction="tighten")
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    token = result.calibration_token

    # First apply: must succeed
    apply_result = apply_calibration(run, db_pool, token)
    assert apply_result.new_version != "v1"

    # Second apply with consumed token: must fail
    svc = _make_calib_service(db_pool)
    with pytest.raises(MemintelError) as exc_info:
        run(svc.apply_calibration(ApplyCalibrationRequest(calibration_token=token)))
    assert "Invalid or expired calibration token" in str(exc_info.value), (
        f"Expected token-consumed error message; got: {exc_info.value}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Apply-calibration edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_apply_calibration_db_state(db_pool, run):
    """
    After apply_calibration(), the DB has TWO versions of the condition.

    Old version is preserved unchanged.  New version carries the calibrated
    params.  No tasks exist so tasks_pending_rebind is empty.
    """
    cid = "dbstate.concept"
    eid = "dbstate.condition"
    concept   = _float_concept(cid, "dbstate.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    result = calibrate(run, db_pool, eid, "v1", feedback_direction="relax")
    apply_result = apply_calibration(run, db_pool, result.calibration_token)
    new_ver = apply_result.new_version

    store = DefinitionStore(db_pool)

    # Both versions must exist in DB
    orig_body = run(store.get(eid, "v1"))
    new_body  = run(store.get(eid, new_ver))
    assert orig_body is not None, "v1 must still exist after apply"
    assert new_body  is not None, f"{new_ver} must exist after apply"

    # Old version unchanged at 0.8
    assert abs(orig_body["strategy"]["params"]["value"] - 0.8) < 0.01, (
        f"v1 threshold must remain 0.8; got {orig_body['strategy']['params']['value']}"
    )

    # New version relaxed (above 0.8 for 'below' + 'relax' → +0.1 = 0.9)
    new_val = new_body["strategy"]["params"]["value"]
    assert new_val > 0.8, f"Relaxed threshold must be > 0.8; got {new_val}"

    # tasks_pending_rebind is empty (no tasks registered)
    assert apply_result.tasks_pending_rebind == [], (
        f"No tasks registered; expected empty list, got {apply_result.tasks_pending_rebind}"
    )


def test_apply_calibration_invalid_token(db_pool, run):
    """
    apply_calibration() with a made-up token → PARAMETER_ERROR (HTTP 400).

    CalibrationTokenStore.resolve_and_invalidate() returns None for unknown
    tokens → MemintelError(PARAMETER_ERROR).
    """
    svc = _make_calib_service(db_pool)
    with pytest.raises(MemintelError) as exc_info:
        run(svc.apply_calibration(ApplyCalibrationRequest(
            calibration_token="this-token-does-not-exist",
        )))
    assert "Invalid or expired calibration token" in str(exc_info.value), (
        f"Expected token error message; got: {exc_info.value}"
    )


def test_apply_calibration_condition_isolation(db_pool, run):
    """
    Calibrating condition A does not affect condition B.

    Two independent conditions are registered.  Only condition A is calibrated
    and applied.  Condition B must remain unchanged in the DB.
    """
    # Condition A
    cid_a = "iso_a.concept"
    eid_a = "iso_a.condition"
    build_condition(
        run, db_pool,
        _float_concept(cid_a, "iso_a.score"),
        _threshold_condition(eid_a, cid_a, "below", 0.8),
    )

    # Condition B
    cid_b = "iso_b.concept"
    eid_b = "iso_b.condition"
    build_condition(
        run, db_pool,
        _float_concept(cid_b, "iso_b.score"),
        _threshold_condition(eid_b, cid_b, "above", 0.5),
    )

    # Calibrate and apply only A
    result = calibrate(run, db_pool, eid_a, "v1", feedback_direction="tighten")
    apply_result = apply_calibration(run, db_pool, result.calibration_token)
    new_ver_a = apply_result.new_version

    store = DefinitionStore(db_pool)

    # A: new version exists
    assert run(store.get(eid_a, new_ver_a)) is not None, (
        f"New version of A ({new_ver_a}) must exist"
    )

    # B: only v1 exists — no extra version created
    body_b_v1 = run(store.get(eid_b, "v1"))
    assert body_b_v1 is not None, "B v1 must still exist"
    assert abs(body_b_v1["strategy"]["params"]["value"] - 0.5) < 0.01, (
        f"B v1 threshold must be unchanged at 0.5; got {body_b_v1['strategy']['params']['value']}"
    )

    # B: no 'v1.1' or any other version
    body_b_v1_1 = run(store.get(eid_b, "v1.1"))
    assert body_b_v1_1 is None, (
        "Calibrating A must not create a new version for B — found unexpected v1.1"
    )


def test_apply_calibration_double_apply(db_pool, run):
    """
    apply_calibration() called twice with the same token — second raises 400.

    This is a repeat of the single-use invariant from section 3, but this
    time verified at the apply level rather than the calibrate level.
    The first apply returns the new version; the second raises PARAMETER_ERROR.
    """
    cid = "double_apply.concept"
    eid = "double_apply.condition"
    build_condition(
        run, db_pool,
        _float_concept(cid, "double_apply.score"),
        _threshold_condition(eid, cid, "below", 0.8),
    )

    result = calibrate(run, db_pool, eid, "v1", feedback_direction="tighten")
    token   = result.calibration_token

    # First apply: success
    apply1 = apply_calibration(run, db_pool, token)
    assert apply1.new_version != "v1", "First apply must produce a new version"

    # Second apply with same token: PARAMETER_ERROR
    svc = _make_calib_service(db_pool)
    with pytest.raises(MemintelError) as exc_info:
        run(svc.apply_calibration(ApplyCalibrationRequest(calibration_token=token)))

    # Verify only ONE new version was created
    store = DefinitionStore(db_pool)
    rows  = run(db_pool.fetch(
        "SELECT version FROM definitions WHERE definition_id = $1 AND definition_type = 'condition'",
        eid,
    ))
    versions = {r["version"] for r in rows}
    assert len(versions) == 2, (  # v1 + one calibrated version
        f"Expected exactly 2 condition versions (v1 + calibrated); got {versions}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Feedback loop + V7 interaction
# ═══════════════════════════════════════════════════════════════════════════════


def test_feedback_loop_task_created_condition(db_pool, run, llm_mock):
    """
    End-to-end: task creation → condition → feedback → calibrate → apply.

    The condition is created via POST /tasks (M5 LLM path).  Feedback is
    submitted against the LLM-generated condition.  The condition is then
    calibrated and applied to produce a new version.

    LLMMockClient routes 'repayment' intent → loan.repayment_below_threshold (v1,
    threshold below 0.8).  After 3 false_negative feedbacks, calibrate → relax
    → new threshold ≈ 0.9.
    """
    app       = _make_full_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Create task → condition auto-registered
            task_resp = await client.post("/tasks", json={
                "intent":       "alert when loan repayment ratio drops below threshold",
                "entity_scope": "loan",
                "delivery": {
                    "type":     "webhook",
                    "endpoint": "https://test.example.com/hook",
                },
                "stream":          False,
                "return_reasoning": False,
            })
            assert task_resp.status_code == 200, (
                f"Task creation failed: {task_resp.status_code}: {task_resp.text}"
            )
            task_data = task_resp.json()
        return task_data

    task_data = run(_go())

    condition_id      = task_data["condition_id"]
    condition_version = task_data["condition_version"]
    concept_id        = task_data["concept_id"]

    assert condition_id, "Task response must include condition_id"
    assert condition_version, "Task response must include condition_version"

    # Step 2: Insert 3 decision records for the task-created condition
    entity = "loan-entity-v7"
    for ts in _TS[:3]:
        insert_decision(
            run, db_pool,
            condition_id, condition_version,
            concept_id, entity,
            fired=False, timestamp_iso=ts,
        )
        resp = submit_feedback(
            run, db_pool,
            condition_id, condition_version,
            entity, ts, "false_negative",
        )
        assert resp.status == "recorded", f"Feedback not recorded for ts={ts}"

    # Step 3: Calibrate → relax (majority false_negative)
    result = calibrate(run, db_pool, condition_id, condition_version)
    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        f"Expected recommendation after 3 false_negative feedbacks; got {result.status}"
    )
    token = result.calibration_token
    assert token is not None

    # Step 4: Apply → new version
    apply_result = apply_calibration(run, db_pool, token)
    new_version  = apply_result.new_version
    assert new_version != condition_version, (
        f"New version must differ from {condition_version}; got {new_version}"
    )

    # Step 5: Verify relaxed threshold in DB (below + relax → +0.1)
    new_body = run(DefinitionStore(db_pool).get(condition_id, new_version))
    assert new_body is not None, f"New version {new_version} not found in DB"
    new_val = new_body["strategy"]["params"]["value"]
    orig_body = run(DefinitionStore(db_pool).get(condition_id, condition_version))
    orig_val  = orig_body["strategy"]["params"]["value"]
    assert new_val > orig_val, (
        f"Relaxed threshold must be greater than original "
        f"(orig={orig_val}, new={new_val})"
    )


def test_calibrate_with_mixed_feedback_tie(db_pool, run):
    """
    Equal counts of false_positive and false_negative → no_recommendation.

    With 3 fp + 3 fn: fp_count == fn_count == 3 — no majority.
    derive_direction() returns None → no_recommendation(insufficient_data).

    Note: The reason label is 'insufficient_data' even though feedback count
    meets the threshold — the code returns insufficient_data whenever
    direction is None (including ties).
    """
    cid = "tie.concept"
    eid = "tie.condition"
    concept   = _float_concept(cid, "tie.score")
    condition = _threshold_condition(eid, cid, "below", 0.8)
    build_condition(run, db_pool, concept, condition)

    # 3 false_positive + 3 false_negative → tie
    entity_fp = "entity-tie-fp"
    entity_fn = "entity-tie-fn"

    for ts in _TS[:3]:
        insert_decision(run, db_pool, eid, "v1", cid, entity_fp, fired=True, timestamp_iso=ts)
        submit_feedback(run, db_pool, eid, "v1", entity_fp, ts, "false_positive")

    for ts in _TS[3:6]:
        insert_decision(run, db_pool, eid, "v1", cid, entity_fn, fired=False, timestamp_iso=ts)
        submit_feedback(run, db_pool, eid, "v1", entity_fn, ts, "false_negative")

    # 6 total records, but tied → no recommendation
    result = calibrate(run, db_pool, eid, "v1")
    assert result.status == CalibrationStatus.NO_RECOMMENDATION, (
        f"Expected no_recommendation on tie vote; got {result.status}"
    )
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA, (
        f"Expected insufficient_data reason (tie); got {result.no_recommendation_reason}"
    )
    assert result.calibration_token is None, "No token expected on no_recommendation"

    # DB unchanged — only v1 exists
    rows = run(db_pool.fetch(
        "SELECT version FROM definitions WHERE definition_id = $1 AND definition_type = 'condition'",
        eid,
    ))
    versions = {r["version"] for r in rows}
    assert versions == {"v1"}, (
        f"No new version must be created after no_recommendation; got {versions}"
    )
