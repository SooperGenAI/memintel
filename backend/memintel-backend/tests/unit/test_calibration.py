"""
tests/unit/test_calibration.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for CalibrationService.

Coverage:
  1. equals strategy      → always no_recommendation (not_applicable_strategy)
  2. composite strategy   → always no_recommendation (not_applicable_strategy)
  3. false_positive majority → tighten direction → recommendation_available
  4. bounds_exceeded + reject policy → no_recommendation (bounds_exceeded)
  5. apply_calibration creates new version; old version unchanged
  6. apply_calibration invalidates token (second use returns error)
  7. tasks_pending_rebind is informational — tasks NOT rebound

Test isolation: every test builds its own stores and service instance.
No shared mutable state between tests.
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.calibration import (
    ApplyCalibrationRequest,
    CalibrationStatus,
    CalibrateRequest,
    FeedbackRecord,
    FeedbackValue,
    NoRecommendationReason,
)
from app.models.concept import DefinitionResponse, SearchResult, VersionSummary
from app.models.condition import StrategyType
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.models.task import DeliveryConfig, DeliveryType, Namespace, Task, TaskStatus
from app.registry.definitions import DefinitionRegistry
from app.services.calibration import CalibrationService


# ── Mock: FeedbackStore ────────────────────────────────────────────────────────

class MockFeedbackStore:
    """
    In-memory FeedbackStore.  Records are added via add() before each test.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], list[FeedbackRecord]] = {}
        self._counter = 0

    def add(
        self,
        condition_id: str,
        condition_version: str,
        feedback: FeedbackValue,
    ) -> None:
        """Add a feedback record directly (bypasses the DB constraint logic)."""
        self._counter += 1
        key = (condition_id, condition_version)
        self._records.setdefault(key, [])
        self._records[key].append(
            FeedbackRecord(
                feedback_id=f"fb-{self._counter}",
                condition_id=condition_id,
                condition_version=condition_version,
                entity=f"entity_{self._counter}",
                timestamp=f"2024-01-0{self._counter}T00:00:00Z",
                feedback=feedback,
                recorded_at=f"2024-01-0{self._counter}T00:00:00Z",
            )
        )

    async def get_by_condition(
        self, condition_id: str, version: str
    ) -> list[FeedbackRecord]:
        return self._records.get((condition_id, version), [])


# ── Mock: CalibrationTokenStore ────────────────────────────────────────────────

class MockCalibrationTokenStore:
    """
    In-memory CalibrationTokenStore.

    create() generates a random token_string and stores the token.
    resolve_and_invalidate() redeems a token exactly once.
    """

    def __init__(self) -> None:
        from app.models.calibration import CalibrationToken
        self._tokens: dict[str, CalibrationToken] = {}
        self._used: set[str] = set()

    async def create(self, token: Any) -> str:
        from app.models.calibration import CalibrationToken
        token_string = secrets.token_urlsafe(16)
        self._tokens[token_string] = CalibrationToken(
            token_string=token_string,
            condition_id=token.condition_id,
            condition_version=token.condition_version,
            recommended_params=token.recommended_params,
            expires_at=datetime(9999, 12, 31, tzinfo=timezone.utc),
        )
        return token_string

    async def resolve_and_invalidate(self, token_string: str) -> Any | None:
        if token_string in self._used or token_string not in self._tokens:
            return None
        self._used.add(token_string)
        return self._tokens[token_string]


# ── Mock: TaskStore ────────────────────────────────────────────────────────────

class MockTaskStore:
    """
    In-memory TaskStore (calibration-relevant subset).
    Pre-populate with tasks via add() before each test.
    """

    def __init__(self) -> None:
        self._tasks: list[Task] = []
        self._updated: list[dict] = []   # records any (would-be) update calls

    def add(self, task: Task) -> None:
        self._tasks.append(task)

    async def find_by_condition_version(
        self, condition_id: str, version: str
    ) -> list[Task]:
        return [
            t for t in self._tasks
            if t.condition_id == condition_id and t.condition_version == version
        ]


# ── Mock: DefinitionStore (backing DefinitionRegistry) ────────────────────────

class MockDefinitionStore:
    """
    In-memory DefinitionStore.  Mirrors the one in test_task_authoring.py.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}
        self._bodies: dict[tuple[str, str], dict] = {}
        self._insert_order: list[tuple[str, str]] = []

    def seed(self, body: dict[str, Any]) -> None:
        """Pre-populate a definition body directly (no validation)."""
        definition_id = (
            body.get("condition_id")
            or body.get("concept_id")
            or body.get("action_id")
            or body.get("id")
            or ""
        )
        version = body.get("version", "")
        key = (definition_id, version)
        ts = datetime.now(timezone.utc)
        self._rows[key] = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type="condition",
            namespace=Namespace(body.get("namespace", "personal")),
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._bodies[key] = body
        self._insert_order.append(key)

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict[str, Any],
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key in self._rows:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' already registered.",
                location=f"{definition_id}:{version}",
            )
        ts = datetime.now(timezone.utc)
        response = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=Namespace(namespace),
            meaning_hash=meaning_hash,
            ir_hash=ir_hash,
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._rows[key] = response
        self._bodies[key] = body
        self._insert_order.append(key)
        return response

    async def get(self, definition_id: str, version: str) -> dict[str, Any] | None:
        return self._bodies.get((definition_id, version))

    async def get_metadata(
        self, definition_id: str, version: str
    ) -> DefinitionResponse | None:
        return self._rows.get((definition_id, version))

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        ordered = [k for k in reversed(self._insert_order) if k[0] == definition_id]
        return [
            VersionSummary(
                version=k[1],
                created_at=self._rows[k].created_at,
                deprecated=self._rows[k].deprecated,
            )
            for k in ordered
        ]

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        items = list(self._rows.values())
        return SearchResult(items=items[:limit], has_more=False, total_count=len(items))

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
        reason: str,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(f"Definition '{definition_id}' version '{version}' not found.")
        updated = self._rows[key].model_copy(
            update={"deprecated": True, "replacement_version": replacement_version}
        )
        self._rows[key] = updated
        return updated

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(f"Definition '{definition_id}' version '{version}' not found.")
        updated = self._rows[key].model_copy(update={"namespace": Namespace(to_namespace)})
        self._rows[key] = updated
        return updated


# ── Mock: GuardrailsStore ──────────────────────────────────────────────────────

class MockGuardrailsStore:
    """
    Minimal duck-typed guardrails store for calibration tests.

    bounds is keyed by strategy name, e.g. {"threshold": {"min": 0.0, "max": 1.0}}.
    on_bounds_exceeded: 'clamp' | 'reject'
    """

    def __init__(
        self,
        bounds: dict[str, dict[str, float | None]] | None = None,
        on_bounds_exceeded: str = "clamp",
    ) -> None:
        self._bounds = bounds or {}
        self._on_bounds_exceeded = on_bounds_exceeded

    def get_guardrails(self) -> Any:
        oob = self._on_bounds_exceeded

        class _Constraints:
            on_bounds_exceeded = oob

        class _Guardrails:
            constraints = _Constraints()

        return _Guardrails()

    def get_threshold_bounds(self, strategy: str) -> dict[str, Any]:
        return self._bounds.get(strategy, {})


# ── Fixture builders ───────────────────────────────────────────────────────────

def _threshold_condition_body(
    condition_id: str = "cond-1",
    version: str = "1.0",
    value: float = 0.8,
    direction: str = "above",
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "version": version,
        "concept_id": "concept-1",
        "concept_version": "1.0",
        "strategy": {
            "type": "threshold",
            "params": {"direction": direction, "value": value},
        },
        "namespace": "personal",
    }


def _equals_condition_body(
    condition_id: str = "cond-eq",
    version: str = "1.0",
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "version": version,
        "concept_id": "concept-1",
        "concept_version": "1.0",
        "strategy": {
            "type": "equals",
            "params": {"value": "HIGH"},
        },
        "namespace": "personal",
    }


def _composite_condition_body(
    condition_id: str = "cond-comp",
    version: str = "1.0",
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "version": version,
        "concept_id": "concept-1",
        "concept_version": "1.0",
        "strategy": {
            "type": "composite",
            "params": {"operator": "AND", "operands": [
                {"condition_id": "cond-a", "condition_version": "1.0"},
                {"condition_id": "cond-b", "condition_version": "1.0"},
            ]},
        },
        "namespace": "personal",
    }


def _make_task(condition_id: str, condition_version: str, task_id: str = "task-1") -> Task:
    return Task(
        task_id=task_id,
        intent="test task",
        concept_id="concept-1",
        concept_version="1.0",
        condition_id=condition_id,
        condition_version=condition_version,
        action_id="action-1",
        action_version="1.0",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack"),
        status=TaskStatus.ACTIVE,
    )


def _make_service(
    feedback_store: MockFeedbackStore | None = None,
    token_store: MockCalibrationTokenStore | None = None,
    task_store: MockTaskStore | None = None,
    def_store: MockDefinitionStore | None = None,
    bounds: dict | None = None,
    on_bounds_exceeded: str = "clamp",
) -> tuple[CalibrationService, MockDefinitionStore]:
    fb   = feedback_store  or MockFeedbackStore()
    tok  = token_store     or MockCalibrationTokenStore()
    ts   = task_store      or MockTaskStore()
    ds   = def_store       or MockDefinitionStore()
    gs   = MockGuardrailsStore(bounds=bounds, on_bounds_exceeded=on_bounds_exceeded)
    registry = DefinitionRegistry(store=ds)
    svc = CalibrationService(
        feedback_store=fb,
        token_store=tok,
        task_store=ts,
        definition_registry=registry,
        guardrails_store=gs,
    )
    return svc, ds


def run(coro: Any) -> Any:
    """Run a coroutine synchronously — mirrors the helper in test_task_authoring.py."""
    return asyncio.run(coro)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_calibrate_equals_always_no_recommendation() -> None:
    """equals strategy → no_recommendation(not_applicable_strategy), regardless of feedback."""
    svc, ds = _make_service()
    ds.seed(_equals_condition_body())

    # Add feedback that would normally trigger a recommendation
    fb = MockFeedbackStore()
    for _ in range(5):
        fb.add("cond-eq", "1.0", FeedbackValue.FALSE_POSITIVE)
    svc._feedback_store = fb

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-eq", condition_version="1.0")))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.NOT_APPLICABLE_STRATEGY
    assert result.calibration_token is None
    assert result.recommended_params is None


def test_calibrate_composite_always_no_recommendation() -> None:
    """composite strategy → no_recommendation(not_applicable_strategy)."""
    svc, ds = _make_service()
    ds.seed(_composite_condition_body())

    fb = MockFeedbackStore()
    for _ in range(5):
        fb.add("cond-comp", "1.0", FeedbackValue.FALSE_POSITIVE)
    svc._feedback_store = fb

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-comp", condition_version="1.0")))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.NOT_APPLICABLE_STRATEGY
    assert result.calibration_token is None


def test_calibrate_false_positive_majority_tightens() -> None:
    """
    A majority of false_positive feedback produces a tighten recommendation.
    For threshold strategy, tighten → increase 'value'.
    """
    fb = MockFeedbackStore()
    # 4 false_positive, 1 false_negative → clear majority → tighten
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)
    fb.add("cond-1", "1.0", FeedbackValue.FALSE_NEGATIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body(value=0.8))

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-1", condition_version="1.0")))

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.calibration_token is not None
    assert result.recommended_params is not None
    # Tighten on threshold → increase value
    assert result.recommended_params["value"] > 0.8
    # current_params reflects original
    assert result.current_params["value"] == 0.8


def test_calibrate_false_negative_majority_relaxes() -> None:
    """
    A majority of false_negative feedback produces a relax recommendation.
    For threshold strategy, relax → decrease 'value'.
    """
    fb = MockFeedbackStore()
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_NEGATIVE)
    fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body(value=0.8))

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-1", condition_version="1.0")))

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params["value"] < 0.8


def test_calibrate_insufficient_feedback_no_recommendation() -> None:
    """Fewer than MIN_FEEDBACK_THRESHOLD records → insufficient_data."""
    fb = MockFeedbackStore()
    # Only 2 records — below threshold of 3
    fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)
    fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body())

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-1", condition_version="1.0")))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA


def test_calibrate_tie_no_recommendation() -> None:
    """Equal false_positive / false_negative counts → no clear direction → insufficient_data."""
    fb = MockFeedbackStore()
    for _ in range(3):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)
    for _ in range(3):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_NEGATIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body())

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-1", condition_version="1.0")))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA


def test_calibrate_bounds_exceeded_reject_policy() -> None:
    """
    When the adjusted value would exceed a bound and on_bounds_exceeded='reject',
    calibrate returns no_recommendation(bounds_exceeded).
    """
    # threshold value = 0.9, tighten would increase it, but max=0.9 → already at bound
    fb = MockFeedbackStore()
    for _ in range(5):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(
        feedback_store=fb,
        bounds={"threshold": {"min": None, "max": 0.9}},
        on_bounds_exceeded="reject",
    )
    ds.seed(_threshold_condition_body(value=0.9))  # value is already at max

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-1", condition_version="1.0")))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.BOUNDS_EXCEEDED
    assert result.calibration_token is None


def test_calibrate_bounds_clamp_policy_succeeds() -> None:
    """
    When the adjusted value would exceed a bound and on_bounds_exceeded='clamp',
    the value is clipped to the bound and a recommendation is returned.
    """
    fb = MockFeedbackStore()
    for _ in range(5):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(
        feedback_store=fb,
        bounds={"threshold": {"min": None, "max": 0.9}},
        on_bounds_exceeded="clamp",
    )
    ds.seed(_threshold_condition_body(value=0.9))

    result = run(svc.calibrate(CalibrateRequest(condition_id="cond-1", condition_version="1.0")))

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    # Clamped to the max bound
    assert result.recommended_params["value"] == 0.9


def test_calibrate_explicit_feedback_direction_skips_aggregation() -> None:
    """
    Explicit feedback_direction overrides stored feedback — no records needed.
    """
    svc, ds = _make_service()   # empty feedback store
    ds.seed(_threshold_condition_body(value=0.8))

    result = run(svc.calibrate(
        CalibrateRequest(
            condition_id="cond-1",
            condition_version="1.0",
            feedback_direction="tighten",
        )
    ))

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params["value"] > 0.8


def test_apply_calibration_creates_new_version_old_unchanged() -> None:
    """
    apply_calibration registers a new condition version with updated params.
    The original version remains in the registry with its original params.
    """
    fb = MockFeedbackStore()
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body(value=0.8))

    # Step 1: get a calibration token
    cal_result = run(svc.calibrate(
        CalibrateRequest(condition_id="cond-1", condition_version="1.0")
    ))
    assert cal_result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    token_str = cal_result.calibration_token
    assert token_str is not None

    # Step 2: apply the calibration
    apply_result = run(svc.apply_calibration(
        ApplyCalibrationRequest(calibration_token=token_str)
    ))

    assert apply_result.condition_id == "cond-1"
    assert apply_result.previous_version == "1.0"
    assert apply_result.new_version == "1.1"       # auto-incremented

    # New version is registered with the recommended params
    new_body = ds._bodies.get(("cond-1", "1.1"))
    assert new_body is not None
    assert new_body["strategy"]["params"]["value"] == apply_result.params_applied["value"]

    # Old version is unchanged
    old_body = ds._bodies.get(("cond-1", "1.0"))
    assert old_body is not None
    assert old_body["strategy"]["params"]["value"] == 0.8


def test_apply_calibration_invalidates_token_second_use_fails() -> None:
    """
    A calibration token can only be used once.
    The second apply_calibration with the same token raises PARAMETER_ERROR.
    """
    fb = MockFeedbackStore()
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body(value=0.8))

    # First use — should succeed
    cal_result = run(svc.calibrate(
        CalibrateRequest(condition_id="cond-1", condition_version="1.0")
    ))
    token_str = cal_result.calibration_token

    run(svc.apply_calibration(ApplyCalibrationRequest(calibration_token=token_str)))

    # Second use with the same token — must fail
    with pytest.raises(MemintelError) as exc_info:
        run(svc.apply_calibration(ApplyCalibrationRequest(calibration_token=token_str)))

    assert exc_info.value.error_type == ErrorType.PARAMETER_ERROR


def test_apply_calibration_tasks_pending_rebind_informational_tasks_not_rebound() -> None:
    """
    tasks_pending_rebind is populated with tasks still on the old version.
    The tasks themselves are NOT updated — they retain their original condition_version.
    """
    fb = MockFeedbackStore()
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    task_store = MockTaskStore()
    task = _make_task(condition_id="cond-1", condition_version="1.0")
    task_store.add(task)

    svc, ds = _make_service(feedback_store=fb, task_store=task_store)
    ds.seed(_threshold_condition_body(value=0.8))

    cal_result = run(svc.calibrate(
        CalibrateRequest(condition_id="cond-1", condition_version="1.0")
    ))
    apply_result = run(svc.apply_calibration(
        ApplyCalibrationRequest(calibration_token=cal_result.calibration_token)
    ))

    # tasks_pending_rebind is populated
    assert len(apply_result.tasks_pending_rebind) == 1
    assert apply_result.tasks_pending_rebind[0].task_id == "task-1"
    assert apply_result.tasks_pending_rebind[0].intent == "test task"

    # The task in the store still has the OLD condition_version
    still_pending = run(task_store.find_by_condition_version("cond-1", "1.0"))
    assert len(still_pending) == 1
    assert still_pending[0].condition_version == "1.0"

    # The task is NOT bound to the new version
    newly_bound = run(task_store.find_by_condition_version("cond-1", "1.1"))
    assert len(newly_bound) == 0


def test_apply_calibration_explicit_new_version() -> None:
    """Caller can supply an explicit new_version instead of auto-increment."""
    fb = MockFeedbackStore()
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body(value=0.8))

    cal_result = run(svc.calibrate(
        CalibrateRequest(condition_id="cond-1", condition_version="1.0")
    ))
    apply_result = run(svc.apply_calibration(
        ApplyCalibrationRequest(
            calibration_token=cal_result.calibration_token,
            new_version="2.0",
        )
    ))

    assert apply_result.new_version == "2.0"
    assert ds._bodies.get(("cond-1", "2.0")) is not None


def test_apply_calibration_duplicate_version_raises_conflict() -> None:
    """
    If new_version already exists in the registry, apply_calibration raises ConflictError.
    """
    fb = MockFeedbackStore()
    for _ in range(4):
        fb.add("cond-1", "1.0", FeedbackValue.FALSE_POSITIVE)

    svc, ds = _make_service(feedback_store=fb)
    ds.seed(_threshold_condition_body(value=0.8))
    # Pre-register the auto-increment target
    ds.seed(_threshold_condition_body(version="1.1", value=0.85))

    cal_result = run(svc.calibrate(
        CalibrateRequest(condition_id="cond-1", condition_version="1.0")
    ))

    with pytest.raises(ConflictError):
        run(svc.apply_calibration(
            ApplyCalibrationRequest(calibration_token=cal_result.calibration_token)
        ))


def test_calibrate_condition_not_found_raises() -> None:
    """calibrate raises NotFoundError when the condition does not exist."""
    svc, _ds = _make_service()

    with pytest.raises(NotFoundError):
        run(svc.calibrate(
            CalibrateRequest(condition_id="nonexistent", condition_version="1.0")
        ))


# ── Unit tests for pure helper methods ────────────────────────────────────────

def test_derive_direction_tighten() -> None:
    svc, _ = _make_service()
    records = [
        FeedbackRecord(
            feedback_id=f"fb-{i}",
            condition_id="c",
            condition_version="1.0",
            entity=f"e{i}",
            timestamp="2024-01-01T00:00:00Z",
            feedback=FeedbackValue.FALSE_POSITIVE,
            recorded_at="2024-01-01T00:00:00Z",
        )
        for i in range(4)
    ]
    assert svc.derive_direction(records) == "tighten"


def test_derive_direction_relax() -> None:
    svc, _ = _make_service()
    records = [
        FeedbackRecord(
            feedback_id=f"fb-{i}",
            condition_id="c",
            condition_version="1.0",
            entity=f"e{i}",
            timestamp="2024-01-01T00:00:00Z",
            feedback=FeedbackValue.FALSE_NEGATIVE,
            recorded_at="2024-01-01T00:00:00Z",
        )
        for i in range(4)
    ]
    assert svc.derive_direction(records) == "relax"


def test_derive_direction_insufficient_data() -> None:
    svc, _ = _make_service()
    records = [
        FeedbackRecord(
            feedback_id="fb-1",
            condition_id="c",
            condition_version="1.0",
            entity="e1",
            timestamp="2024-01-01T00:00:00Z",
            feedback=FeedbackValue.FALSE_POSITIVE,
            recorded_at="2024-01-01T00:00:00Z",
        )
    ]
    assert svc.derive_direction(records) is None


def test_derive_direction_tie_returns_none() -> None:
    svc, _ = _make_service()
    records = [
        FeedbackRecord(
            feedback_id=f"fp-{i}",
            condition_id="c",
            condition_version="1.0",
            entity=f"e{i}",
            timestamp="2024-01-01T00:00:00Z",
            feedback=FeedbackValue.FALSE_POSITIVE,
            recorded_at="2024-01-01T00:00:00Z",
        )
        for i in range(3)
    ] + [
        FeedbackRecord(
            feedback_id=f"fn-{i}",
            condition_id="c",
            condition_version="1.0",
            entity=f"en{i}",
            timestamp="2024-01-01T00:00:00Z",
            feedback=FeedbackValue.FALSE_NEGATIVE,
            recorded_at="2024-01-01T00:00:00Z",
        )
        for i in range(3)
    ]
    assert svc.derive_direction(records) is None


def test_auto_increment_version_minor() -> None:
    svc, _ = _make_service()
    assert svc._auto_increment_version("1.0") == "1.1"
    assert svc._auto_increment_version("1.9") == "1.10"
    assert svc._auto_increment_version("2.3") == "2.4"


def test_auto_increment_version_single_component() -> None:
    svc, _ = _make_service()
    assert svc._auto_increment_version("1") == "2"
    assert svc._auto_increment_version("9") == "10"


def test_adjust_params_threshold_tighten_increases_value() -> None:
    svc, _ = _make_service()
    result = svc.adjust_params(
        strategy=StrategyType.THRESHOLD,
        current_params={"direction": "above", "value": 1.0},
        direction="tighten",
        target=None,
        bounds={},
        on_bounds_exceeded="clamp",
    )
    assert result is not None
    assert result["value"] > 1.0


def test_adjust_params_threshold_relax_decreases_value() -> None:
    svc, _ = _make_service()
    result = svc.adjust_params(
        strategy=StrategyType.THRESHOLD,
        current_params={"direction": "above", "value": 1.0},
        direction="relax",
        target=None,
        bounds={},
        on_bounds_exceeded="clamp",
    )
    assert result is not None
    assert result["value"] < 1.0


def test_adjust_params_percentile_relax_increases_value() -> None:
    svc, _ = _make_service()
    result = svc.adjust_params(
        strategy=StrategyType.PERCENTILE,
        current_params={"direction": "top", "value": 10.0},
        direction="relax",
        target=None,
        bounds={},
        on_bounds_exceeded="clamp",
    )
    assert result is not None
    assert result["value"] > 10.0


def test_adjust_params_z_score_tighten_increases_threshold() -> None:
    svc, _ = _make_service()
    result = svc.adjust_params(
        strategy=StrategyType.Z_SCORE,
        current_params={"threshold": 2.0, "direction": "above", "window": "30d"},
        direction="tighten",
        target=None,
        bounds={},
        on_bounds_exceeded="clamp",
    )
    assert result is not None
    assert result["threshold"] > 2.0


def test_adjust_params_reject_on_max_exceeded() -> None:
    svc, _ = _make_service()
    result = svc.adjust_params(
        strategy=StrategyType.THRESHOLD,
        current_params={"direction": "above", "value": 1.0},
        direction="tighten",
        target=None,
        bounds={"max": 1.0},
        on_bounds_exceeded="reject",
    )
    assert result is None


def test_adjust_params_clamp_on_max_exceeded() -> None:
    svc, _ = _make_service()
    result = svc.adjust_params(
        strategy=StrategyType.THRESHOLD,
        current_params={"direction": "above", "value": 1.0},
        direction="tighten",
        target=None,
        bounds={"max": 1.0},
        on_bounds_exceeded="clamp",
    )
    assert result is not None
    assert result["value"] == 1.0   # clamped to max
