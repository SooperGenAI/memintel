"""
tests/unit/test_feedback.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for FeedbackService.

All tests run synchronously via asyncio.run() — no pytest-asyncio required.

Coverage:
  1. Invalid feedback value ('useful') → parameter_error BEFORE any DB call.
  2. Decision not found (condition not in registry) → NotFoundError (404).
  3. Duplicate feedback submission → ConflictError (409).
  4. Valid submission succeeds → FeedbackResponse(status='recorded').
  5. First submission succeeds; second raises ConflictError.
"""
from __future__ import annotations

import asyncio

import pytest

from app.models.calibration import (
    FeedbackRecord,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackValue,
)
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.services.feedback import FeedbackService


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def _make_request(**overrides):
    """Build a FeedbackRequest with sensible defaults."""
    defaults = dict(
        condition_id="org.churn_risk",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
        feedback=FeedbackValue.FALSE_POSITIVE,
        note=None,
    )
    defaults.update(overrides)
    return FeedbackRequest(**defaults)


def _make_record(
    *,
    feedback_id: str = "fb-001",
    condition_id: str = "org.churn_risk",
    condition_version: str = "1.0",
    entity: str = "user_42",
    timestamp: str = "2024-01-15T09:00:00Z",
    feedback: FeedbackValue = FeedbackValue.FALSE_POSITIVE,
    note: str | None = None,
) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=feedback_id,
        condition_id=condition_id,
        condition_version=condition_version,
        entity=entity,
        timestamp=timestamp,
        feedback=feedback,
        note=note,
        recorded_at="2024-01-15T10:00:00Z",
    )


# ── Mock stores ───────────────────────────────────────────────────────────────

class MockFeedbackStore:
    """In-memory FeedbackStore for unit tests."""

    def __init__(self):
        # uniqueness key → FeedbackRecord
        self._records: dict[tuple, FeedbackRecord] = {}
        self.create_calls: list[FeedbackRecord] = []

    async def create(self, record: FeedbackRecord) -> FeedbackRecord:
        self.create_calls.append(record)
        key = (
            record.condition_id,
            record.condition_version,
            record.entity,
            record.timestamp,
        )
        if key in self._records:
            raise ConflictError(
                "Feedback already submitted for this decision.",
                location="(condition_id, condition_version, entity, timestamp)",
            )
        stored = FeedbackRecord(
            feedback_id=f"fb-{len(self._records) + 1:03d}",
            condition_id=record.condition_id,
            condition_version=record.condition_version,
            entity=record.entity,
            timestamp=record.timestamp,
            feedback=record.feedback,
            note=record.note,
            recorded_at="2024-01-15T10:00:00Z",
        )
        self._records[key] = stored
        return stored


class MockDefinitionRegistry:
    """Registry stub that raises NotFoundError for unknown (id, version) pairs."""

    def __init__(self, known: set[tuple] | None = None):
        # Set of (condition_id, condition_version) that are "registered".
        self._known: set[tuple] = known or set()
        self.get_calls: list[tuple] = []

    def seed(self, condition_id: str, version: str) -> None:
        self._known.add((condition_id, version))

    async def get(self, definition_id: str, version: str) -> dict:
        self.get_calls.append((definition_id, version))
        key = (definition_id, version)
        if key not in self._known:
            raise NotFoundError(
                f"Condition '{definition_id}' version '{version}' not found.",
            )
        # Return a minimal condition body — FeedbackService doesn't parse it.
        return {"condition_id": definition_id, "version": version}


def _make_service(
    *,
    condition_registered: bool = True,
    condition_id: str = "org.churn_risk",
    condition_version: str = "1.0",
) -> tuple[FeedbackService, MockFeedbackStore, MockDefinitionRegistry]:
    registry = MockDefinitionRegistry()
    if condition_registered:
        registry.seed(condition_id, condition_version)
    fb_store = MockFeedbackStore()
    svc = FeedbackService(feedback_store=fb_store, definition_registry=registry)
    return svc, fb_store, registry


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_invalid_feedback_value_raises_parameter_error_before_db():
    """
    'useful' is not a valid FeedbackValue.
    PARAMETER_ERROR must be raised BEFORE the registry or feedback store
    is touched — verified by asserting neither mock was called.
    """
    svc, fb_store, registry = _make_service()

    # Bypass pydantic validation to deliver an invalid string to the service.
    req = FeedbackRequest.model_construct(
        condition_id="org.churn_risk",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
        feedback="useful",
        note=None,
    )

    with pytest.raises(MemintelError) as exc_info:
        run(svc.submit(req))

    assert exc_info.value.error_type == ErrorType.PARAMETER_ERROR
    assert "useful" in str(exc_info.value).lower()

    # Neither the registry nor the feedback store must have been called.
    assert registry.get_calls == [], "registry.get() must not be called on invalid input"
    assert fb_store.create_calls == [], "feedback_store.create() must not be called on invalid input"


def test_not_useful_also_rejected_before_db():
    """'not_useful' is another invalid alias — same parameter_error guarantee."""
    svc, fb_store, registry = _make_service()

    req = FeedbackRequest.model_construct(
        condition_id="org.churn_risk",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
        feedback="not_useful",
        note=None,
    )

    with pytest.raises(MemintelError) as exc_info:
        run(svc.submit(req))

    assert exc_info.value.error_type == ErrorType.PARAMETER_ERROR
    assert registry.get_calls == []
    assert fb_store.create_calls == []


def test_decision_not_found_raises_not_found_error():
    """
    When the condition is not in the registry, FeedbackService must raise
    NotFoundError (→ HTTP 404).
    """
    svc, fb_store, registry = _make_service(condition_registered=False)
    req = _make_request()

    with pytest.raises(NotFoundError):
        run(svc.submit(req))

    # Registry was consulted but feedback store must not be called.
    assert len(registry.get_calls) == 1
    assert fb_store.create_calls == []


def test_duplicate_submission_raises_conflict():
    """
    FeedbackStore raises ConflictError on duplicate (same uniqueness key).
    FeedbackService must propagate it without modification → HTTP 409.
    """
    svc, fb_store, registry = _make_service()
    req = _make_request()

    # First submission succeeds.
    resp = run(svc.submit(req))
    assert resp.status == "recorded"

    # Second submission for the exact same decision → ConflictError.
    with pytest.raises(ConflictError):
        run(svc.submit(req))

    assert len(fb_store.create_calls) == 2


def test_valid_submission_returns_recorded_response():
    """
    Valid feedback is stored; FeedbackResponse.status == 'recorded' and
    feedback_id is the DB-assigned identifier from the store.
    """
    svc, fb_store, registry = _make_service()
    req = _make_request()

    resp = run(svc.submit(req))

    assert isinstance(resp, FeedbackResponse)
    assert resp.status == "recorded"
    assert resp.feedback_id.startswith("fb-")

    # Exactly one record was written.
    assert len(fb_store.create_calls) == 1
    written = fb_store.create_calls[0]
    assert written.condition_id == req.condition_id
    assert written.condition_version == req.condition_version
    assert written.entity == req.entity
    assert written.feedback == FeedbackValue.FALSE_POSITIVE


def test_note_field_stored_correctly():
    """Note field is stored as-is (may contain PII — never logged)."""
    svc, fb_store, _ = _make_service()
    req = _make_request(
        feedback=FeedbackValue.CORRECT,
        note="Looked legitimate — customer confirmed correct alert.",
    )

    resp = run(svc.submit(req))
    assert resp.status == "recorded"

    written = fb_store.create_calls[0]
    assert written.note == req.note


def test_all_valid_feedback_values_accepted():
    """false_positive, false_negative, and correct are all accepted."""
    for fv in FeedbackValue:
        svc, fb_store, _ = _make_service()
        req = _make_request(feedback=fv, entity=f"user_{fv.value}")
        resp = run(svc.submit(req))
        assert resp.status == "recorded", f"Expected 'recorded' for {fv}"
