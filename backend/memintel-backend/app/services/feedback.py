"""
app/services/feedback.py
──────────────────────────────────────────────────────────────────────────────
FeedbackService — feedback ingestion and validation.

Validates that a decision record exists (condition registered) before persisting
feedback. Delegates storage to FeedbackStore, which enforces the uniqueness
constraint at both the application and DB layer.

Pipeline for submit():
  1. Validate feedback_value — reject anything that is not false_positive,
     false_negative, or correct BEFORE any store call.
  2. Verify condition is registered (decision could have been made).
     Raises NotFoundError → HTTP 404 if condition not found.
  3. Build FeedbackRecord and call FeedbackStore.create().
     FeedbackStore raises ConflictError → HTTP 409 on duplicate.
  4. Return FeedbackResponse(status='recorded', feedback_id=<DB-assigned id>).

PII note
────────
The `note` field may contain free-text PII. It is stored as-is but MUST
NEVER be logged — not here, not in the store, not anywhere.
"""
from __future__ import annotations

from typing import Any

import structlog

from app.models.calibration import (
    FeedbackRecord,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackValue,
)
from app.models.errors import ErrorType, MemintelError, NotFoundError

log = structlog.get_logger(__name__)


# Valid feedback value strings — checked explicitly before any DB call.
_VALID_FEEDBACK: frozenset[str] = frozenset(v.value for v in FeedbackValue)


class FeedbackService:
    """
    Validates and persists decision feedback records.

    submit() validates the feedback value, verifies a decision exists
    (condition registered at the given version), then delegates to
    FeedbackStore.create(). Raises NotFoundError or ConflictError as
    appropriate — never catches them.

    PII: The `note` field may contain PII. Never log it.

    Parameters
    ──────────
    feedback_store      — must implement async create(record) → FeedbackRecord.
                          Raises ConflictError on uniqueness key violation.
    definition_registry — must implement async get(id, version) → dict.
                          Raises NotFoundError if the condition is not registered.
    decision_store      — must implement async find_by_condition_entity_timestamp(
                          condition_id, condition_version, entity_id, timestamp)
                          → DecisionRecord | None. Used to verify the decision exists.
    """

    def __init__(
        self,
        feedback_store: Any,
        definition_registry: Any,
        decision_store: Any,
    ) -> None:
        self._feedback_store = feedback_store
        self._registry = definition_registry
        self._decision_store = decision_store

    # ── Public API ──────────────────────────────────────────────────────────────

    async def submit(self, req: FeedbackRequest) -> FeedbackResponse:
        """
        Validate and persist decision feedback.

        Raises:
          MemintelError(PARAMETER_ERROR) — feedback value is not one of:
            false_positive, false_negative, correct.
            'useful' and 'not_useful' are explicitly rejected.
            This check happens BEFORE any store call.
          NotFoundError — condition (id, version) not found in registry.
          ConflictError — feedback already exists for this (condition_id,
            condition_version, entity, timestamp) uniqueness key.
        """
        # Step 1 — validate feedback value BEFORE any DB call.
        _validate_feedback_value(req.feedback)

        # Step 2 — verify condition is registered (decision could have been made).
        #  Raises NotFoundError → HTTP 404 if not found.
        await self._registry.get(req.condition_id, req.condition_version)

        # Step 3 — verify a decision record exists for (condition_id, condition_version,
        #  entity, timestamp). Raises NotFoundError → HTTP 404 if not found.
        decision = await self._decision_store.find_by_condition_entity_timestamp(
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            entity_id=req.entity,
            timestamp=req.timestamp,
        )
        if decision is None:
            raise NotFoundError(
                f"No decision record found for condition '{req.condition_id}' "
                f"version '{req.condition_version}', entity '{req.entity}', "
                f"timestamp '{req.timestamp}'.",
                location="timestamp",
                suggestion="Provide the exact evaluated_at timestamp from the decision record.",
            )

        # Step 4 — persist the feedback record.
        #  FeedbackStore.create() raises ConflictError → HTTP 409 on duplicate.
        record = await self._feedback_store.create(
            FeedbackRecord(
                feedback_id="",          # DB-assigned; placeholder — store overwrites
                condition_id=req.condition_id,
                condition_version=req.condition_version,
                entity=req.entity,
                timestamp=req.timestamp,
                feedback=req.feedback,
                note=req.note,
                recorded_at="",          # DB-assigned; placeholder — store overwrites
            )
        )

        log.info(
            "feedback_submitted",
            feedback_id=record.feedback_id,
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            entity=req.entity,
            feedback=(
                req.feedback.value
                if isinstance(req.feedback, FeedbackValue)
                else str(req.feedback)
            ),
            # note intentionally omitted — may contain PII
        )

        return FeedbackResponse(status="recorded", feedback_id=record.feedback_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_feedback_value(feedback: Any) -> None:
    """
    Raise PARAMETER_ERROR if feedback is not a valid FeedbackValue.

    Accepts FeedbackValue enum members (always valid) and raw strings that
    match a valid enum value. Rejects anything else — including 'useful' and
    'not_useful', which are common incorrect aliases.

    Called BEFORE any store access so the DB is never touched on invalid input.
    """
    if isinstance(feedback, FeedbackValue):
        return  # valid enum instance — all good

    raw = str(feedback)
    if raw not in _VALID_FEEDBACK:
        raise MemintelError(
            ErrorType.PARAMETER_ERROR,
            f"Invalid feedback value '{raw}'. "
            "Must be one of: false_positive, false_negative, correct. "
            "'useful' and 'not_useful' are not valid feedback values.",
            location="feedback",
            suggestion="Use FeedbackValue.FALSE_POSITIVE, FALSE_NEGATIVE, or CORRECT.",
        )
