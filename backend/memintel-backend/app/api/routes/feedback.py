"""
app/api/routes/feedback.py
──────────────────────────────────────────────────────────────────────────────
Feedback ingestion endpoint.

Endpoints
─────────
  POST /feedback/decision    submitFeedback — record feedback on a decision

Ownership rules
───────────────
This endpoint is deterministic — no LLM involvement.

POST /feedback/decision:
  Records user feedback on a specific decision result. Feedback is stored as
  a structured record and later consumed by POST /conditions/calibrate to
  derive the tighten/relax direction for parameter adjustment.

  Valid feedback values:
    false_positive — condition fired but should not have → tighten on calibrate
    false_negative — condition did not fire but should have → relax on calibrate
    correct        — expected behaviour → no calibration adjustment

  Validation:
    A decision must exist for (condition_id, condition_version, entity,
    timestamp). HTTP 404 if no such decision record exists.

  Deduplication:
    One feedback record per decision — uniqueness key:
      (condition_id, condition_version, entity, timestamp)
    Duplicate submissions → HTTP 409. Rationale: duplicates bias calibration
    direction counts and degrade recommendation quality.

  PII note:
    The `note` field may contain free-text PII. It is stored as-is but is
    NEVER logged — not in this route, not in the service, not in the store.
    Structured log fields intentionally omit `note`.

Error handling
──────────────
NotFoundError  → HTTP 404 (raised by FeedbackService when decision not found).
ConflictError  → HTTP 409 (raised by FeedbackStore on unique key violation).
MemintelError subclasses are caught globally by the exception handler in main.py.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends

from app.api.deps import require_api_key
from app.models.calibration import FeedbackRequest, FeedbackResponse
from app.persistence.db import get_db
from app.registry.definitions import DefinitionRegistry
from app.services.feedback import FeedbackService
from app.stores.decision import DecisionStore
from app.stores.definition import DefinitionStore
from app.stores.feedback import FeedbackStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_feedback_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> FeedbackService:
    """
    FastAPI dependency — returns a FeedbackService backed by the shared pool.

    FeedbackService validates that a decision record exists for the submitted
    (condition_id, condition_version, entity, timestamp) tuple, then persists
    the feedback record. It raises NotFoundError or ConflictError as needed.
    """
    feedback_store = FeedbackStore(pool)
    definition_store = DefinitionStore(pool)
    definition_registry = DefinitionRegistry(store=definition_store)
    decision_store = DecisionStore(pool)
    return FeedbackService(
        feedback_store=feedback_store,
        definition_registry=definition_registry,
        decision_store=decision_store,
    )


# ── POST /feedback/decision ────────────────────────────────────────────────────

@router.post(
    "/decision",
    summary="Submit feedback on a decision",
    response_model=FeedbackResponse,
    status_code=200,
)
async def submit_feedback(
    req: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service),
    _: None = Depends(require_api_key),
) -> FeedbackResponse:
    """
    Record user feedback on a specific decision result.

    Feedback is an input to the calibration loop:
      false_positive → signals the condition is too sensitive (tighten)
      false_negative → signals the condition is too strict (relax)
      correct        → no adjustment needed

    Feedback on an equals strategy condition is stored and accepted, but
    POST /conditions/calibrate will return no_recommendation for it (equals
    has no numeric parameter to adjust).

    Returns FeedbackResponse with status='recorded' and the assigned feedback_id.
    The note field is stored as-is but is never echoed back in any log output.

    HTTP 404 — no decision record found for (condition_id, condition_version,
               entity, timestamp). Feedback must reference a real decision.
    HTTP 409 — feedback already submitted for this decision (duplicate).
    """
    # Log without the note field — it may contain PII.
    log.info(
        "submit_feedback_request",
        extra={
            "condition_id": req.condition_id,
            "condition_version": req.condition_version,
            "feedback": req.feedback.value,
        },
    )
    return await service.submit(req)
