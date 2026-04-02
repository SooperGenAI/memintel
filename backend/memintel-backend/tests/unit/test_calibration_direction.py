"""
tests/unit/test_calibration_direction.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the direction-aware threshold calibration logic in
CalibrationService.adjust_params().

Background
──────────
The threshold strategy fires in two modes:
  direction="above"  → fires when concept_value > threshold
  direction="below"  → fires when concept_value < threshold

Correct adjustment semantics per mode:
  above:
    tighten (false_positive majority) → RAISE threshold
      fewer values exceed a higher bar → fewer firings
    relax (false_negative majority)   → LOWER threshold
      more values exceed a lower bar  → more firings

  below:
    tighten (false_positive majority) → LOWER threshold
      fewer values fall below a lower bar → fewer firings
    relax (false_negative majority)   → RAISE threshold
      more values fall below a higher bar → more firings

Strategies other than threshold (z_score, percentile, change) are not
direction-aware in the same sense — see inline notes in each test.

All six tests call CalibrationService.calibrate() end-to-end through the
mock stack so the full pipeline (feedback aggregation → direction derivation
→ adjust_params → token creation) is exercised, not just the helper.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.models.calibration import (
    CalibrateRequest,
    CalibrationStatus,
    FeedbackValue,
)
from app.models.condition import StrategyType

# Re-use the mock stack and helpers from the existing unit test module.
from tests.unit.test_calibration import (
    MockFeedbackStore,
    _make_service,
    _threshold_condition_body,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── Helper ─────────────────────────────────────────────────────────────────────


def _calibrate(
    condition_body: dict,
    feedback_val: FeedbackValue,
    n_records: int = 3,
    bounds: dict | None = None,
    on_bounds_exceeded: str = "clamp",
):
    """
    Seed n_records feedback records, calibrate, and return the result.

    Uses the mock stack — no DB, no HTTP.
    """
    fb = MockFeedbackStore()
    cid = condition_body["condition_id"]
    cver = condition_body["version"]
    for _ in range(n_records):
        fb.add(cid, cver, feedback_val)

    svc, ds = _make_service(
        feedback_store=fb,
        bounds=bounds,
        on_bounds_exceeded=on_bounds_exceeded,
    )
    ds.seed(condition_body)

    return run(svc.calibrate(CalibrateRequest(condition_id=cid, condition_version=cver)))


# ══════════════════════════════════════════════════════════════════════════════
# 1 — below / false_positive → tighten → LOWER threshold
# ══════════════════════════════════════════════════════════════════════════════


def test_below_false_positive_lowers_threshold() -> None:
    """
    direction="below", threshold=0.35, 3 × false_positive
    → tighten → LOWER value to 0.25.

    "below" fires when concept_value < threshold.
    False positives mean it fires too often; to tighten, we lower the threshold
    so fewer values fall below it.

    step = max(0.35 * 0.1, 0.1) = 0.1
    expected: 0.35 - 0.1 = 0.25
    """
    result = _calibrate(
        condition_body=_threshold_condition_body(value=0.35, direction="below"),
        feedback_val=FeedbackValue.FALSE_POSITIVE,
        n_records=3,
    )

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert result.recommended_params["value"] < 0.35, (
        f"below/false_positive → tighten → must lower threshold below 0.35. "
        f"Got {result.recommended_params['value']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2 — below / false_negative → relax → RAISE threshold
# ══════════════════════════════════════════════════════════════════════════════


def test_below_false_negative_raises_threshold() -> None:
    """
    direction="below", threshold=0.35, 3 × false_negative
    → relax → RAISE value to 0.45.

    False negatives mean it misses too often; to relax, we raise the threshold
    so more values fall below it.

    step = max(0.35 * 0.1, 0.1) = 0.1
    expected: 0.35 + 0.1 = 0.45
    """
    result = _calibrate(
        condition_body=_threshold_condition_body(value=0.35, direction="below"),
        feedback_val=FeedbackValue.FALSE_NEGATIVE,
        n_records=3,
    )

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert result.recommended_params["value"] > 0.35, (
        f"below/false_negative → relax → must raise threshold above 0.35. "
        f"Got {result.recommended_params['value']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3 — above / false_positive → tighten → RAISE threshold
# ══════════════════════════════════════════════════════════════════════════════


def test_above_false_positive_raises_threshold() -> None:
    """
    direction="above", threshold=0.35, 3 × false_positive
    → tighten → RAISE value.

    "above" fires when concept_value > threshold.
    False positives mean it fires too often; to tighten, we raise the threshold
    so fewer values exceed it.

    step = max(0.35 * 0.1, 0.1) = 0.1
    expected: 0.35 + 0.1 = 0.45
    """
    result = _calibrate(
        condition_body=_threshold_condition_body(value=0.35, direction="above"),
        feedback_val=FeedbackValue.FALSE_POSITIVE,
        n_records=3,
    )

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert result.recommended_params["value"] > 0.35, (
        f"above/false_positive → tighten → must raise threshold above 0.35. "
        f"Got {result.recommended_params['value']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4 — above / false_negative → relax → LOWER threshold
# ══════════════════════════════════════════════════════════════════════════════


def test_above_false_negative_lowers_threshold() -> None:
    """
    direction="above", threshold=0.35, 3 × false_negative
    → relax → LOWER value.

    False negatives mean it misses too often; to relax, we lower the threshold
    so more values exceed it.

    step = max(0.35 * 0.1, 0.1) = 0.1
    expected: 0.35 - 0.1 = 0.25
    """
    result = _calibrate(
        condition_body=_threshold_condition_body(value=0.35, direction="above"),
        feedback_val=FeedbackValue.FALSE_NEGATIVE,
        n_records=3,
    )

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert result.recommended_params["value"] < 0.35, (
        f"above/false_negative → relax → must lower threshold below 0.35. "
        f"Got {result.recommended_params['value']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5 — below / tighten near lower bound → clamped to 0.0
# ══════════════════════════════════════════════════════════════════════════════


def test_below_adjustment_clamped_to_lower_bound() -> None:
    """
    direction="below", threshold=0.05, 3 × false_positive, bounds [0.0, 1.0].

    Tighten/below → lower threshold:
      step = max(0.05 * 0.1, 0.1) = 0.1
      raw = 0.05 - 0.1 = -0.05
      clamped to min=0.0 → recommended = 0.0

    Result must be >= 0.0 (lower bound respected).
    """
    result = _calibrate(
        condition_body=_threshold_condition_body(value=0.05, direction="below"),
        feedback_val=FeedbackValue.FALSE_POSITIVE,
        n_records=3,
        bounds={"threshold": {"min": 0.0, "max": 1.0}},
        on_bounds_exceeded="clamp",
    )

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert result.recommended_params["value"] >= 0.0, (
        f"Clamped adjustment must respect lower bound 0.0. "
        f"Got {result.recommended_params['value']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6 — above / tighten near upper bound → clamped to 1.0
# ══════════════════════════════════════════════════════════════════════════════


def test_above_adjustment_clamped_to_upper_bound() -> None:
    """
    direction="above", threshold=0.95, 3 × false_positive, bounds [0.0, 1.0].

    Tighten/above → raise threshold:
      step = max(0.95 * 0.1, 0.1) = max(0.095, 0.1) = 0.1
      raw = 0.95 + 0.1 = 1.05
      clamped to max=1.0 → recommended = 1.0

    Result must be <= 1.0 (upper bound respected).
    """
    result = _calibrate(
        condition_body=_threshold_condition_body(value=0.95, direction="above"),
        feedback_val=FeedbackValue.FALSE_POSITIVE,
        n_records=3,
        bounds={"threshold": {"min": 0.0, "max": 1.0}},
        on_bounds_exceeded="clamp",
    )

    assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert result.recommended_params is not None
    assert result.recommended_params["value"] <= 1.0, (
        f"Clamped adjustment must respect upper bound 1.0. "
        f"Got {result.recommended_params['value']}"
    )
