"""
app/api/routes/conditions.py
──────────────────────────────────────────────────────────────────────────────
Condition lifecycle endpoints — inspection, explanation, calibration, and
application of calibration recommendations.

Endpoints
─────────
  GET  /conditions/{id}                getCondition        — fetch by id + version
  POST /conditions/explain             explainCondition    — explain logic and params
  POST /conditions/calibrate           calibrateCondition  — generate recommendation
  POST /conditions/apply-calibration   applyCalibration    — apply rec as new version

Ownership rules
───────────────
All four endpoints are deterministic — no LLM involvement.

GET /conditions/{id}:
  version is REQUIRED as a query parameter. Returns the ConditionDefinition
  stored in the definitions table. HTTP 404 if (condition_id, version) not found.

POST /conditions/explain:
  Returns a human-readable explanation of what a condition evaluates and
  why its parameters were chosen. Delegates to ExplanationService which
  derives explanations deterministically from the strategy type and params.
  No LLM. HTTP 404 if condition not found.

POST /conditions/calibrate:
  Analyses stored feedback and/or a target alert volume to produce a
  parameter recommendation. Returns HTTP 200 always — inspect status field
  for 'recommendation_available' vs 'no_recommendation'.
  Returns no_recommendation for equals strategy (no numeric param to adjust)
  or when insufficient feedback data exists.
  Does NOT modify the existing condition. HTTP 404 if condition not found.

POST /conditions/apply-calibration:
  Consumes a single-use calibration_token (from calibrate response) and
  creates a new immutable condition version with the recommended params.
  Does NOT automatically rebind any tasks — rebinding is an explicit user
  action via PATCH /tasks/{id}.
  Returns tasks_pending_rebind (informational; callers must rebind manually).
  HTTP 400 if the token is invalid, expired, or already used.

Route registration order
────────────────────────
Literal-path routes (/explain, /calibrate, /apply-calibration) are defined
BEFORE the parameterised route (/{id}) to ensure FastAPI matches them
correctly. In practice there is no ambiguity (different HTTP methods), but
the ordering is maintained as a defensive convention.

Error handling
──────────────
MemintelError subclasses are caught globally by the exception handler in
main.py — routes do not catch them here.
HTTP status codes mirror developer_api.yaml.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends, Query
from fastapi import Request

from app.models.calibration import (
    ApplyCalibrationRequest,
    ApplyCalibrationResult,
    CalibrateRequest,
    CalibrationResult,
)
from app.models.condition import ConditionDefinition, ConditionExplanation
from app.models.errors import NotFoundError
from app.persistence.db import get_db
from app.persistence.stores import get_definition_store
from app.services.calibration import CalibrationService
from app.services.explanation import ExplanationService
from app.registry.definitions import DefinitionRegistry
from app.stores import (
    CalibrationTokenStore,
    ContextStore,
    DefinitionStore,
    FeedbackStore,
    TaskStore,
)
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/conditions", tags=["Conditions"])


# ── Request body for POST /conditions/explain ──────────────────────────────────
# Inline schema from developer_api.yaml (not a named component).

class ExplainConditionRequest(BaseModel):
    condition_id: str = Field(..., max_length=255)
    condition_version: str = Field(..., max_length=50)
    timestamp: str | None = None    # ISO 8601 UTC — optional, for context only


# ── Service dependencies ───────────────────────────────────────────────────────

async def get_explanation_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> ExplanationService:
    """
    FastAPI dependency — returns an ExplanationService backed by the shared pool.

    ExplanationService generates human-readable condition explanations
    deterministically from strategy type and parameters. No LLM involvement.
    """
    definition_registry = DefinitionRegistry(store=DefinitionStore(pool))
    return ExplanationService(
        definition_registry=definition_registry,
        concept_executor=None,
        condition_evaluator=None,
        data_resolver=None,
    )


async def get_calibration_service(
    pool: asyncpg.Pool = Depends(get_db),
    request: Request = None,
) -> CalibrationService:
    """
    FastAPI dependency — returns a CalibrationService wired to the pool and
    the guardrails_store from app.state.

    CalibrationService is the only service that reads threshold_bounds from
    the guardrails and computes parameter adjustments. It is fully
    deterministic — no LLM involvement.
    """
    guardrails_store = request.app.state.guardrails_store
    return CalibrationService(
        feedback_store=FeedbackStore(pool),
        token_store=CalibrationTokenStore(pool),
        task_store=TaskStore(pool),
        definition_registry=DefinitionRegistry(store=DefinitionStore(pool)),
        guardrails_store=guardrails_store,
        context_store=ContextStore(pool),
    )


# ── POST /conditions/explain ──────────────────────────────────────────────────
# Defined before /{id} to avoid any path-matching ambiguity.

@router.post(
    "/explain",
    summary="Explain condition logic and parameters",
    response_model=ConditionExplanation,
    status_code=200,
)
async def explain_condition(
    req: ExplainConditionRequest,
    service: ExplanationService = Depends(get_explanation_service),
) -> ConditionExplanation:
    """
    Return a human-readable explanation of a condition definition.

    Explains what the condition evaluates, why its parameters were selected,
    and its relationship to its concept. Fully deterministic — no LLM.

    HTTP 404 — condition (id, version) not found.
    """
    log.info(
        "explain_condition_request",
        extra={
            "condition_id": req.condition_id,
            "condition_version": req.condition_version,
        },
    )
    return await service.explain_condition(
        condition_id=req.condition_id,
        condition_version=req.condition_version,
        timestamp=req.timestamp,
    )


# ── POST /conditions/calibrate ────────────────────────────────────────────────

@router.post(
    "/calibrate",
    summary="Generate calibration recommendation for a condition",
    response_model=CalibrationResult,
    status_code=200,
)
async def calibrate_condition(
    req: CalibrateRequest,
    service: CalibrationService = Depends(get_calibration_service),
) -> CalibrationResult:
    """
    Analyse stored feedback and/or a target alert volume to recommend
    adjusted parameters.

    Always returns HTTP 200 — inspect the status field:
      recommendation_available — calibration_token + recommended_params populated.
      no_recommendation        — no_recommendation_reason populated.

    Returns no_recommendation when:
      - Strategy is 'equals' (no numeric parameter to adjust).
      - Insufficient feedback records (fewer than MIN_FEEDBACK_THRESHOLD).
      - Adjustment would violate guardrail bounds and on_bounds_exceeded='reject'.

    Does NOT modify the existing condition. Call POST /conditions/apply-calibration
    with the returned calibration_token to create a new version.

    HTTP 404 — condition (id, version) not found.
    """
    log.info(
        "calibrate_condition_request",
        extra={
            "condition_id": req.condition_id,
            "condition_version": req.condition_version,
        },
    )
    return await service.calibrate(req)


# ── POST /conditions/apply-calibration ────────────────────────────────────────

@router.post(
    "/apply-calibration",
    summary="Apply a calibration recommendation as a new condition version",
    response_model=ApplyCalibrationResult,
    status_code=200,
)
async def apply_calibration(
    req: ApplyCalibrationRequest,
    service: CalibrationService = Depends(get_calibration_service),
) -> ApplyCalibrationResult:
    """
    Consume a calibration_token and create a new immutable condition version
    using the recommended parameters stored in the token.

    The token is single-use and expires 24 hours after generation. Calling
    this endpoint invalidates the token atomically — concurrent calls with
    the same token return HTTP 400 for all but the first.

    Does NOT automatically rebind tasks. tasks_pending_rebind in the response
    lists tasks still referencing the previous condition version — the caller
    must rebind them explicitly via PATCH /tasks/{id}.

    new_version is optional. When absent, the service auto-increments the
    source version (e.g. '1.0' → '1.1').

    HTTP 400 — calibration_token is invalid, expired, or already used.
    """
    log.info("apply_calibration_request")
    return await service.apply_calibration(req)


# ── GET /conditions/{id} ──────────────────────────────────────────────────────
# Parameterised route last — after all literal-path routes.

@router.get(
    "/{condition_id}",
    summary="Get a condition definition",
    response_model=ConditionDefinition,
    status_code=200,
)
async def get_condition(
    condition_id: str,
    version: str = Query(
        ...,
        description="Condition version to retrieve (required)",
    ),
    store: DefinitionStore = Depends(get_definition_store),
) -> ConditionDefinition:
    """
    Return the full ConditionDefinition for the given (condition_id, version).

    Includes strategy type, parameters, concept binding (concept_id +
    concept_version), namespace, deprecation status, and creation timestamp.

    Use to inspect condition logic and parameters before explaining or calibrating.

    HTTP 404 — condition not found or version does not exist.
    """
    body = await store.get(condition_id, version)
    if body is None:
        raise NotFoundError(
            f"Condition '{condition_id}' version '{version}' not found.",
            location="condition_id",
        )
    return ConditionDefinition.model_validate(body)
