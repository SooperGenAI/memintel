"""
tests/integration/test_calibration_edge_cases.py
──────────────────────────────────────────────────────────────────────────────
CalibrationService edge-case tests — all in-process, no DB, LLM, or HTTP.

Coverage
────────
  1. test_zero_feedback_no_crash
       feedback_store returns [] → no_recommendation(insufficient_data),
       no division-by-zero, no token created.

  2. test_all_false_positive_recommends_tighten
       Majority false_positive → direction='tighten' → threshold 'value'
       increases (harder to trigger → fewer alerts).

  3. test_all_false_negative_recommends_relax
       Majority false_negative → direction='relax' → threshold 'value'
       decreases (easier to trigger → more alerts).

  4. test_apply_calibration_registers_before_querying_tasks
       Call ordering: registry.register() is called BEFORE
       task_store.find_by_condition_version(). The new version is
       immutably persisted before informational task lookup.

  5a. test_concurrent_calibration_two_tokens_issued  (GAP)
       Two concurrent calibrate() calls for the same condition, each with
       sufficient feedback, each independently creates its own token.
       The service issues TWO tokens — there is no per-condition-version
       locking that would collapse concurrent requests into one token.

  5b. test_concurrent_calibration_explicit_direction_safe
       Two concurrent calibrate() calls using explicit feedback_direction
       (no DB feedback read needed). Both succeed and the token strings
       are distinct.

All tests use asyncio.run() to drive async service methods from
synchronous pytest functions, matching the pattern in other integration
test files in this suite.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.models.calibration import (
    ApplyCalibrationRequest,
    CalibrateRequest,
    CalibrationStatus,
    CalibrationToken,
    FeedbackRecord,
    FeedbackValue,
    MIN_FEEDBACK_THRESHOLD,
    NoRecommendationReason,
)
from app.models.condition import StrategyType
from app.models.errors import MemintelError, ErrorType, NotFoundError
from app.services.calibration import CalibrationService


# ── Shared fixtures ────────────────────────────────────────────────────────────

#: A threshold condition body that ConditionDefinition.model_validate() accepts.
_THRESHOLD_BODY: dict = {
    "condition_id": "org.churn_alert",
    "version": "1.0",
    "concept_id": "org.churn_risk_score",
    "concept_version": "1.0",
    "strategy": {
        "type": "threshold",
        "params": {"direction": "above", "value": 0.7},
    },
    "namespace": "org",
}

#: A z_score condition body (param key is 'threshold', not 'value').
_ZSCORE_BODY: dict = {
    "condition_id": "org.payment_anomaly",
    "version": "1.0",
    "concept_id": "org.payment_failure_rate",
    "concept_version": "1.0",
    "strategy": {
        "type": "z_score",
        "params": {"threshold": 2.0, "direction": "above", "window": "30d"},
    },
    "namespace": "org",
}


def _make_feedback(
    feedback_value: FeedbackValue,
    count: int,
    condition_id: str = "org.churn_alert",
    condition_version: str = "1.0",
) -> list[FeedbackRecord]:
    """Return `count` feedback records with the given feedback value."""
    return [
        FeedbackRecord(
            feedback_id=f"fb-{i:04d}",
            condition_id=condition_id,
            condition_version=condition_version,
            entity=f"entity-{i}",
            timestamp="2024-06-01T09:00:00Z",
            feedback=feedback_value,
            recorded_at="2024-06-01T10:00:00Z",
        )
        for i in range(count)
    ]


def _build_service(
    *,
    registry_get_side_effect=None,
    registry_get_return=None,
    feedback_return: list | None = None,
    token_create_return: str = "tok_test_abc",
    token_resolve_return: CalibrationToken | None = None,
    task_find_return: list | None = None,
    bounds: dict | None = None,
    on_exceeded: str = "clamp",
) -> tuple[CalibrationService, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Construct a CalibrationService with fully mocked stores.

    Returns (service, mock_registry, mock_feedback_store, mock_token_store,
             mock_task_store).
    """
    mock_registry    = MagicMock()
    mock_feedback    = MagicMock()
    mock_token_store = MagicMock()
    mock_task_store  = MagicMock()
    mock_guardrails  = MagicMock()

    # registry.get() — supports either a side_effect or a plain return value
    if registry_get_side_effect is not None:
        mock_registry.get = AsyncMock(side_effect=registry_get_side_effect)
    elif registry_get_return is not None:
        mock_registry.get = AsyncMock(return_value=registry_get_return)
    mock_registry.register = AsyncMock(return_value=None)

    # feedback_store.get_by_condition()
    mock_feedback.get_by_condition = AsyncMock(
        return_value=(feedback_return or [])
    )

    # token_store.create() and resolve_and_invalidate()
    mock_token_store.create = AsyncMock(return_value=token_create_return)
    mock_token_store.resolve_and_invalidate = AsyncMock(
        return_value=token_resolve_return
    )

    # task_store.find_by_condition_version()
    mock_task_store.find_by_condition_version = AsyncMock(
        return_value=(task_find_return or [])
    )

    # guardrails_store.get_threshold_bounds() and .get_guardrails()
    mock_guardrails.get_threshold_bounds = MagicMock(
        return_value=(bounds or {})
    )
    guardrails_obj = MagicMock()
    guardrails_obj.constraints.on_bounds_exceeded = on_exceeded
    mock_guardrails.get_guardrails = MagicMock(return_value=guardrails_obj)

    service = CalibrationService(
        feedback_store=mock_feedback,
        token_store=mock_token_store,
        task_store=mock_task_store,
        definition_registry=mock_registry,
        guardrails_store=mock_guardrails,
    )
    return service, mock_registry, mock_feedback, mock_token_store, mock_task_store


# ── Scenario 1: Zero feedback records ─────────────────────────────────────────

def test_zero_feedback_no_crash() -> None:
    """
    When feedback_store returns an empty list, CalibrationService.calibrate()
    must return no_recommendation(insufficient_data) — not crash with a
    division-by-zero, AttributeError, or any unhandled exception.

    Verified behaviours:
      - result.status == CalibrationStatus.NO_RECOMMENDATION
      - result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA
      - result.calibration_token is None  (no token issued for empty data)
      - token_store.create() was NOT called
    """
    service, _, _, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=[],    # zero records
    )

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
    )
    result = asyncio.run(service.calibrate(req))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION, (
        f"Expected NO_RECOMMENDATION, got {result.status}"
    )
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA, (
        f"Expected INSUFFICIENT_DATA, got {result.no_recommendation_reason}"
    )
    assert result.calibration_token is None, (
        "A token must NOT be issued when there are insufficient feedback records"
    )
    mock_token_store.create.assert_not_called()


def test_below_min_threshold_no_crash() -> None:
    """
    MIN_FEEDBACK_THRESHOLD - 1 records (below the threshold) must also
    return no_recommendation(insufficient_data), not crash.

    Ensures the boundary condition (exactly threshold-1 records) is clean.
    """
    records = _make_feedback(FeedbackValue.FALSE_POSITIVE, MIN_FEEDBACK_THRESHOLD - 1)
    service, _, _, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=records,
    )

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
    )
    result = asyncio.run(service.calibrate(req))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA
    mock_token_store.create.assert_not_called()


# ── Scenario 2: All false_positive → tighten ─────────────────────────────────

def test_all_false_positive_recommends_tighten() -> None:
    """
    All feedback records are false_positive.

    false_positive majority → direction='tighten'.

    Threshold bias semantics (py-instructions.md §CalibrationService):
      tighten → INCREASE 'value'  (makes the condition harder to trigger)

    Assertions:
      - result.status == RECOMMENDATION_AVAILABLE
      - recommended_params['value'] > current_params['value']
      - impact.direction == 'decrease'  (fewer alerts after tightening)
      - result.calibration_token is not None
    """
    records = _make_feedback(
        FeedbackValue.FALSE_POSITIVE,
        count=MIN_FEEDBACK_THRESHOLD + 2,   # well above threshold
    )
    service, _, _, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=records,
        token_create_return="tok_tighten_abc",
    )

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
    )
    result = asyncio.run(service.calibrate(req))

    current_val = _THRESHOLD_BODY["strategy"]["params"]["value"]   # 0.7

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        f"Expected RECOMMENDATION_AVAILABLE, got {result.status}"
    )
    assert result.recommended_params is not None
    assert result.recommended_params["value"] > current_val, (
        f"False-positive majority must tighten (increase) the threshold: "
        f"recommended={result.recommended_params['value']}, current={current_val}"
    )
    assert result.impact is not None
    assert result.impact.direction.value == "decrease", (
        "Tightening must estimate a decrease in alert volume"
    )
    assert result.calibration_token == "tok_tighten_abc"
    mock_token_store.create.assert_called_once()


def test_all_false_positive_zscore_tightens_threshold_key() -> None:
    """
    For a z_score strategy (param key = 'threshold', not 'value'),
    all false_positive feedback → tighten → z_score threshold INCREASES.

    Ensures the correct param key is adjusted for z_score.
    """
    records = _make_feedback(
        FeedbackValue.FALSE_POSITIVE,
        count=MIN_FEEDBACK_THRESHOLD,
        condition_id="org.payment_anomaly",
    )
    service, _, _, _, _ = _build_service(
        registry_get_return=_ZSCORE_BODY,
        feedback_return=records,
    )

    req = CalibrateRequest(
        condition_id="org.payment_anomaly",
        condition_version="1.0",
    )
    result = asyncio.run(service.calibrate(req))

    current_threshold = _ZSCORE_BODY["strategy"]["params"]["threshold"]   # 2.0

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert "threshold" in result.recommended_params, (
        "z_score calibration must adjust the 'threshold' key, not 'value'"
    )
    assert result.recommended_params["threshold"] > current_threshold, (
        f"False-positive majority must tighten (increase) z_score threshold: "
        f"recommended={result.recommended_params['threshold']}, "
        f"current={current_threshold}"
    )


# ── Scenario 3: All false_negative → relax ───────────────────────────────────

def test_all_false_negative_recommends_relax() -> None:
    """
    All feedback records are false_negative.

    false_negative majority → direction='relax'.

    Threshold bias semantics (py-instructions.md §CalibrationService):
      relax → DECREASE 'value'  (makes the condition easier to trigger)

    Assertions:
      - result.status == RECOMMENDATION_AVAILABLE
      - recommended_params['value'] < current_params['value']
      - impact.direction == 'increase'  (more alerts after relaxing)
      - result.calibration_token is not None
    """
    records = _make_feedback(
        FeedbackValue.FALSE_NEGATIVE,
        count=MIN_FEEDBACK_THRESHOLD + 2,
    )
    service, _, _, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=records,
        token_create_return="tok_relax_abc",
    )

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
    )
    result = asyncio.run(service.calibrate(req))

    current_val = _THRESHOLD_BODY["strategy"]["params"]["value"]   # 0.7

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE, (
        f"Expected RECOMMENDATION_AVAILABLE, got {result.status}"
    )
    assert result.recommended_params is not None
    assert result.recommended_params["value"] < current_val, (
        f"False-negative majority must relax (decrease) the threshold: "
        f"recommended={result.recommended_params['value']}, current={current_val}"
    )
    assert result.impact is not None
    assert result.impact.direction.value == "increase", (
        "Relaxing must estimate an increase in alert volume"
    )
    assert result.calibration_token == "tok_relax_abc"
    mock_token_store.create.assert_called_once()


def test_mixed_feedback_tie_returns_no_recommendation() -> None:
    """
    Equal counts of false_positive and false_negative (a tie) must return
    no_recommendation(insufficient_data).

    The derive_direction() tie rule: fp_count == fn_count → return None.
    """
    tie_count = MIN_FEEDBACK_THRESHOLD  # exactly at threshold but tied
    fp = _make_feedback(FeedbackValue.FALSE_POSITIVE, tie_count)
    fn = _make_feedback(FeedbackValue.FALSE_NEGATIVE, tie_count)

    service, _, _, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=fp + fn,
    )

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
    )
    result = asyncio.run(service.calibrate(req))

    assert result.status == CalibrationStatus.NO_RECOMMENDATION
    assert result.no_recommendation_reason == NoRecommendationReason.INSUFFICIENT_DATA, (
        "A tie in fp/fn counts must be treated as insufficient directional signal"
    )
    mock_token_store.create.assert_not_called()


def test_opposite_directions_ordered_correctly() -> None:
    """
    Sanity check: running scenario 2 and scenario 3 against the same
    condition base produces opposite recommended values.

    fp_result.recommended_params['value'] > fn_result.recommended_params['value']

    This validates the directional asymmetry: tighten and relax are true
    opposites and the service does not conflate them.
    """
    fp_records = _make_feedback(FeedbackValue.FALSE_POSITIVE, MIN_FEEDBACK_THRESHOLD)
    fn_records = _make_feedback(FeedbackValue.FALSE_NEGATIVE, MIN_FEEDBACK_THRESHOLD)

    svc_fp, _, _, _, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=fp_records,
    )
    svc_fn, _, _, _, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=fn_records,
    )

    req = CalibrateRequest(condition_id="org.churn_alert", condition_version="1.0")

    fp_result = asyncio.run(svc_fp.calibrate(req))
    fn_result = asyncio.run(svc_fn.calibrate(req))

    assert fp_result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert fn_result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE

    fp_val = fp_result.recommended_params["value"]
    fn_val = fn_result.recommended_params["value"]

    assert fp_val > fn_val, (
        f"Tighten (fp) must produce a higher threshold than relax (fn): "
        f"tighten={fp_val}, relax={fn_val}"
    )


# ── Scenario 4: apply_calibration — registry write precedes task lookup ───────

def test_apply_calibration_registers_before_querying_tasks() -> None:
    """
    apply_calibration() must register the new condition version in the
    registry (immutable write) BEFORE querying task_store for pending rebinds.

    The tasks_pending_rebind list is informational only — it is populated
    after the version is safely persisted. A crash during the task query
    must not leave the registry write incomplete.

    Verified by tracking call order via a shared list appended to by
    mock side_effects.

    Also verifies:
      - registry.register() receives the correct new body (version bumped,
        strategy params replaced, source body not mutated).
      - result.previous_version == '1.0'
      - result.new_version == '1.1'  (auto-increment)
      - result.params_applied == recommended_params from the token
    """
    recommended = {"direction": "above", "value": 0.85}
    token = CalibrationToken(
        token_string="tok_apply_test",
        condition_id="org.churn_alert",
        condition_version="1.0",
        recommended_params=recommended,
        expires_at=datetime.now(tz=timezone.utc) + timedelta(hours=1),
    )

    # Track call order using a shared list
    call_order: list[str] = []

    async def _registry_get(cond_id: str, version: str) -> dict:
        if version == "1.0":
            return dict(_THRESHOLD_BODY)
        # New version doesn't exist yet → raises NotFoundError (expected)
        raise NotFoundError(
            f"Condition '{cond_id}' version '{version}' not found.",
            location=f"{cond_id}:{version}",
        )

    async def _registry_register(body: dict, namespace: str, definition_type: str) -> None:
        call_order.append("register")

    async def _task_find(cond_id: str, version: str) -> list:
        call_order.append("find_tasks")
        return []

    service, mock_registry, _, mock_token, mock_tasks = _build_service(
        registry_get_side_effect=_registry_get,
        token_resolve_return=token,
    )
    mock_registry.register   = AsyncMock(side_effect=_registry_register)
    mock_tasks.find_by_condition_version = AsyncMock(side_effect=_task_find)

    req = ApplyCalibrationRequest(calibration_token="tok_apply_test")
    result = asyncio.run(service.apply_calibration(req))

    # ── Ordering: register must precede find_tasks ─────────────────────────────
    assert "register" in call_order, "registry.register() was never called"
    assert "find_tasks" in call_order, "task_store.find_by_condition_version() was never called"
    assert call_order.index("register") < call_order.index("find_tasks"), (
        "registry.register() must be called BEFORE find_by_condition_version(): "
        f"call order was {call_order}"
    )

    # ── Result correctness ─────────────────────────────────────────────────────
    assert result.condition_id == "org.churn_alert"
    assert result.previous_version == "1.0"
    assert result.new_version == "1.1", (
        f"Auto-incremented version must be '1.1', got '{result.new_version}'"
    )
    assert result.params_applied == recommended, (
        f"params_applied must match the token's recommended_params: "
        f"expected {recommended}, got {result.params_applied}"
    )

    # ── Registered body correctness ───────────────────────────────────────────
    register_call_kwargs = mock_registry.register.call_args
    registered_body = register_call_kwargs.args[0]

    assert registered_body["version"] == "1.1", (
        "Registered body must carry the new version"
    )
    assert registered_body["strategy"]["params"] == recommended, (
        "Registered body must carry the recommended params"
    )
    # Source body must not be mutated
    assert _THRESHOLD_BODY["strategy"]["params"]["value"] == 0.7, (
        "apply_calibration() must not mutate the source condition body"
    )


def test_apply_calibration_source_not_mutated() -> None:
    """
    apply_calibration() deep-copies the source body before modifying it.

    The source version in the registry must be identical before and after
    the call — immutability invariant.
    """
    import copy

    recommended = {"direction": "above", "value": 0.95}
    source_body = copy.deepcopy(_THRESHOLD_BODY)   # snapshot before call

    token = CalibrationToken(
        token_string="tok_mutate_test",
        condition_id="org.churn_alert",
        condition_version="1.0",
        recommended_params=recommended,
        expires_at=datetime.now(tz=timezone.utc) + timedelta(hours=1),
    )

    async def _registry_get(cond_id: str, version: str) -> dict:
        if version == "1.0":
            return dict(_THRESHOLD_BODY)
        raise NotFoundError(
            f"Condition '{cond_id}' version '{version}' not found.",
            location=f"{cond_id}:{version}",
        )

    service, _, _, _, _ = _build_service(
        registry_get_side_effect=_registry_get,
        token_resolve_return=token,
    )

    req = ApplyCalibrationRequest(calibration_token="tok_mutate_test")
    asyncio.run(service.apply_calibration(req))

    assert _THRESHOLD_BODY["strategy"]["params"]["value"] == source_body["strategy"]["params"]["value"], (
        "apply_calibration() mutated the source body — deep copy is required"
    )
    assert _THRESHOLD_BODY["version"] == source_body["version"], (
        "apply_calibration() mutated the source body version — deep copy is required"
    )


# ── Scenario 5: Concurrent calibration ───────────────────────────────────────

def test_concurrent_calibration_two_tokens_issued_gap() -> None:
    """
    DOCUMENTED GAP: Two concurrent calibrate() calls for the same condition
    with the same feedback both succeed and each creates its own token.

    CalibrationService.calibrate() has NO per-condition-version lock.
    When two requests arrive simultaneously:
      - Both read the condition from the registry.
      - Both read feedback and derive direction independently.
      - Both compute adjusted params (identically, given the same inputs).
      - Both call token_store.create() — two tokens are issued.

    This test confirms the observed behaviour: token_store.create() is
    called twice. A correct implementation would either:
      a) Use an idempotency key / upsert so concurrent calls for the same
         (condition_id, condition_version) return the SAME token, OR
      b) Serialize concurrent calibrations with a distributed lock.

    If this test starts failing (create called only once), the gap has
    been fixed — update the assertion to assert_called_once().
    """
    records = _make_feedback(FeedbackValue.FALSE_POSITIVE, MIN_FEEDBACK_THRESHOLD)

    # Use unique token strings per call to confirm two distinct tokens are created.
    call_count = 0

    async def _create_token(tok_obj) -> str:
        nonlocal call_count
        call_count += 1
        return f"tok_concurrent_{call_count:02d}"

    service, _, _, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
        feedback_return=records,
    )
    mock_token_store.create = AsyncMock(side_effect=_create_token)

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
    )

    async def run_concurrent() -> list:
        return await asyncio.gather(
            service.calibrate(req),
            service.calibrate(req),
        )

    results = asyncio.run(run_concurrent())

    # Both calls succeed (no error raised by the service)
    assert len(results) == 2
    assert all(r.status == CalibrationStatus.RECOMMENDATION_AVAILABLE for r in results), (
        f"Both concurrent calls must succeed: {[r.status for r in results]}"
    )

    # GAP: token_store.create() is called twice — one token per request.
    assert mock_token_store.create.call_count == 2, (
        "GAP CONFIRMED: two concurrent calibrate() calls issued "
        f"{mock_token_store.create.call_count} token(s). "
        "Expected 2 (no deduplication). "
        "If this assertion fails, per-condition locking has been added — "
        "update this test to assert_called_once()."
    )

    # The two tokens are distinct strings.
    tokens = [r.calibration_token for r in results]
    assert tokens[0] != tokens[1], (
        "The two concurrently issued tokens must be different strings"
    )


def test_concurrent_calibration_explicit_direction_distinct_tokens() -> None:
    """
    Two concurrent calibrate() calls with explicit feedback_direction='tighten'
    both produce recommendation_available results with distinct token strings.

    explicit feedback_direction bypasses the feedback store read entirely —
    verifies that this path is also free of shared-state corruption under
    concurrent access.
    """
    call_n = 0

    async def _create_token(tok_obj) -> str:
        nonlocal call_n
        call_n += 1
        return f"tok_explicit_{call_n:02d}"

    service, _, mock_feedback, mock_token_store, _ = _build_service(
        registry_get_return=_THRESHOLD_BODY,
    )
    mock_token_store.create = AsyncMock(side_effect=_create_token)

    req = CalibrateRequest(
        condition_id="org.churn_alert",
        condition_version="1.0",
        feedback_direction="tighten",   # explicit override — no feedback store read
    )

    async def run_concurrent() -> list:
        return await asyncio.gather(
            service.calibrate(req),
            service.calibrate(req),
        )

    results = asyncio.run(run_concurrent())

    # feedback_store.get_by_condition() must NOT have been called
    mock_feedback.get_by_condition.assert_not_called()

    assert all(r.status == CalibrationStatus.RECOMMENDATION_AVAILABLE for r in results)
    tokens = [r.calibration_token for r in results]
    assert tokens[0] != tokens[1], (
        "Each concurrent request must receive its own distinct token"
    )
