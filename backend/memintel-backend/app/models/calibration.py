"""
app/models/calibration.py
──────────────────────────────────────────────────────────────────────────────
Calibration and feedback domain models.

Covers the full calibration loop:
  - FeedbackValue / FeedbackRecord / FeedbackRequest  — feedback ingestion
  - CalibrateRequest / CalibrationResult              — POST /conditions/calibrate
  - ApplyCalibrationRequest / ApplyCalibrationResult  — POST /conditions/apply-calibration
  - CalibrationToken                                  — single-use token (DB row + wire)
  - CalibrationTelemetry                              — opt-in platform analytics

Design notes
────────────
CalibrationResult.status drives branching, not no_recommendation_reason.
  When status='no_recommendation', reason is always populated.
  When status='recommendation_available', reason is always None.
  This is enforced by the model validator here.

CalibrationToken is used both as the DB row model (includes id, used_at,
  created_at) and as the wire payload embedded in CalibrationResult. The DB
  fields (id, used_at, created_at) are excluded from API serialisation via
  Field(exclude=True). The token_string is the only identifier exposed to
  callers.

ApplyCalibrationRequest carries ONLY calibration_token + new_version.
  Do NOT add condition_id, condition_version, or explicit params. The token
  is the single path — adding fallbacks would break the atomicity guarantee.

tasks_pending_rebind in ApplyCalibrationResult is informational ONLY.
  The service MUST NOT rebind tasks automatically. Callers rebind via
  PATCH /tasks/{id}.

FeedbackRecord.timestamp is a str (ISO 8601 UTC) matching the decision
  timestamp. It corresponds to decision_timestamp in the DB schema. The
  rename exists because 'timestamp' is clearer at the Python API boundary.

MIN_FEEDBACK_THRESHOLD — minimum number of feedback records required before
  CalibrationService.derive_direction() returns a direction. Below this
  threshold, direction=None → status='no_recommendation' with
  reason='insufficient_data'. Defined here as a module constant rather than
  hardcoded in the service.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Calibration threshold constant ────────────────────────────────────────────

#: Minimum number of feedback records required before CalibrationService can
#: derive a direction. Fewer records → 'insufficient_data'.
MIN_FEEDBACK_THRESHOLD: int = 3


# ── Enums ─────────────────────────────────────────────────────────────────────

class ImpactDirection(str, Enum):
    """Direction of estimated alert volume change after calibration."""
    INCREASE  = "increase"
    DECREASE  = "decrease"
    NO_CHANGE = "no_change"


class FeedbackValue(str, Enum):
    """
    Valid feedback values for a decision.

    false_positive — condition fired but should not have → tighten
    false_negative — condition did not fire but should have → relax
    correct        — expected behaviour → no calibration adjustment

    Invalid aliases that must be rejected:
      'useful', 'not_useful' — these are NOT valid values.
    """
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    CORRECT        = "correct"


class CalibrationStatus(str, Enum):
    """
    Outcome of a calibration request.

    recommendation_available — adjusted params computed; calibration_token populated.
    no_recommendation        — no adjustment possible; no_recommendation_reason populated.
    """
    RECOMMENDATION_AVAILABLE = "recommendation_available"
    NO_RECOMMENDATION        = "no_recommendation"


class NoRecommendationReason(str, Enum):
    """
    Reason why CalibrationService returned no recommendation.

    bounds_exceeded          — adjustment would violate guardrail bounds (on_bounds_exceeded='reject')
    not_applicable_strategy  — equals strategy has no numeric parameter to adjust
    insufficient_data        — fewer than MIN_FEEDBACK_THRESHOLD feedback records
    """
    BOUNDS_EXCEEDED         = "bounds_exceeded"
    NOT_APPLICABLE_STRATEGY = "not_applicable_strategy"
    INSUFFICIENT_DATA       = "insufficient_data"


# ── Feedback models ────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """
    Request body for POST /feedback/decision.

    timestamp is the ISO 8601 UTC timestamp of the original decision.
    A decision record must exist for (condition_id, condition_version,
    entity, timestamp) — the route handler raises HTTP 404 if not found.

    note is free-text, may contain PII — never log it.
    """
    condition_id: str
    condition_version: str
    entity: str
    timestamp: str           # ISO 8601 UTC — timestamp of the original decision
    feedback: FeedbackValue
    note: str | None = None


class FeedbackRecord(BaseModel):
    """
    Persisted feedback record. Returned by POST /feedback/decision.

    feedback_id    — DB-assigned UUID (gen_random_uuid()::text).
    timestamp      — maps to decision_timestamp in the DB schema.
    recorded_at    — ISO 8601 UTC; when this feedback was submitted.

    Uniqueness key (DB UNIQUE constraint):
      (condition_id, condition_version, entity, timestamp)
    Duplicate submissions raise ConflictError → HTTP 409.
    """
    feedback_id: str
    condition_id: str
    condition_version: str
    entity: str
    timestamp: str           # timestamp of the original decision
    feedback: FeedbackValue
    note: str | None = None
    recorded_at: str         # ISO 8601 UTC — when this feedback was submitted


# ── FeedbackResponse ──────────────────────────────────────────────────────────

class FeedbackResponse(BaseModel):
    """
    Response body for POST /feedback/decision (HTTP 200).

    status is always 'recorded' — branch on HTTP status codes for errors.
    feedback_id is the DB-assigned UUID for the new record.
    """
    status: str                  # always "recorded"
    feedback_id: str


# ── Calibration token ──────────────────────────────────────────────────────────

class CalibrationToken(BaseModel):
    """
    Single-use, 24-hour-expiry token linking a calibration recommendation to
    its recommended parameters.

    token_string is the opaque random string exposed to callers.
    recommended_params is a generic dict — strategy-specific parameter key/value
    pairs (e.g. {'value': 0.85} for threshold, {'threshold': 2.1} for z_score).

    Internal DB fields (excluded from API serialisation):
      id         — BIGSERIAL primary key; DB-assigned; never in responses.
      used_at    — set atomically on redemption; None until consumed.
      created_at — DB default NOW().

    Expiry and atomicity:
      Tokens expire 24 hours after creation (expires_at).
      resolve_and_invalidate() sets used_at atomically via UPDATE...WHERE used_at IS NULL.
      A None return from resolve_and_invalidate() means expired, already used, or not found.
    """
    token_string: str
    condition_id: str
    condition_version: str
    recommended_params: dict[str, Any]
    expires_at: datetime

    # Internal DB fields — excluded from API serialisation
    id: int | None = Field(default=None, exclude=True)
    used_at: datetime | None = Field(default=None, exclude=True)
    created_at: datetime | None = Field(default=None, exclude=True)

    model_config = {"populate_by_name": True}

    def is_expired(self) -> bool:
        """Return True if this token has passed its expiry time."""
        from datetime import timezone
        return datetime.now(tz=timezone.utc) >= self.expires_at.replace(
            tzinfo=self.expires_at.tzinfo or timezone.utc
        )


# ── Target configuration ───────────────────────────────────────────────────────

class TargetConfig(BaseModel):
    """
    Optional alert volume target for calibration.

    When provided, CalibrationService.adjust_params() targets this alert
    rate rather than applying a fixed step in the feedback direction.

    alerts_per_day is per-entity (not total across all entities).
    """
    alerts_per_day: float


# ── Calibration impact estimate ────────────────────────────────────────────────

class CalibrationImpact(BaseModel):
    """
    Estimated impact of applying the recommended parameter change.

    delta_alerts — estimated change in daily alert volume.
      Positive = more alerts (relax); negative = fewer alerts (tighten).

    direction — qualitative direction of the impact.
    """
    delta_alerts: float
    direction: ImpactDirection


# ── CalibrateRequest / CalibrationResult ──────────────────────────────────────

class CalibrateRequest(BaseModel):
    """
    Request body for POST /conditions/calibrate.

    feedback_direction overrides the direction derived from stored feedback.
    Valid values: 'tighten' | 'relax'. When absent, CalibrationService
    derives direction from the majority of stored feedback records.

    target is optional. When provided, the service targets the specified
    alert volume rather than a fixed adjustment step.
    """
    condition_id: str
    condition_version: str
    target: TargetConfig | None = None
    feedback_direction: str | None = None   # 'tighten' | 'relax' — explicit override


class CalibrationResult(BaseModel):
    """
    Output of POST /conditions/calibrate.

    status drives branching:
      recommendation_available — calibration_token and recommended_params are populated.
      no_recommendation        — no_recommendation_reason is populated; token is None.

    current_params reflects the strategy params of the requested condition version.
    It is always populated regardless of status.

    Invariants (enforced by validator):
      When status=recommendation_available:
        calibration_token, recommended_params, and impact must be non-None.
        no_recommendation_reason must be None.
      When status=no_recommendation:
        no_recommendation_reason must be non-None.
        calibration_token, recommended_params, and impact must be None.
    """
    status: CalibrationStatus
    current_params: dict[str, Any]

    # Populated when status=recommendation_available
    calibration_token: str | None = None
    recommended_params: dict[str, Any] | None = None
    impact: CalibrationImpact | None = None

    # Populated when status=no_recommendation
    no_recommendation_reason: NoRecommendationReason | None = None

    # Context bias adjustment metadata (populated when an active context with
    # calibration_bias caused an adjustment to the statistically optimal value)
    statistically_optimal: float | None = None    # raw value from adjust_params
    context_adjusted: float | None = None         # bias-shifted value (None if no adjustment)
    adjustment_explanation: str | None = None     # human-readable description of adjustment

    @model_validator(mode="after")
    def _check_status_consistency(self) -> CalibrationResult:
        if self.status == CalibrationStatus.RECOMMENDATION_AVAILABLE:
            if self.calibration_token is None:
                raise ValueError(
                    "calibration_token is required when status='recommendation_available'"
                )
            if self.recommended_params is None:
                raise ValueError(
                    "recommended_params is required when status='recommendation_available'"
                )
            if self.no_recommendation_reason is not None:
                raise ValueError(
                    "no_recommendation_reason must be None when status='recommendation_available'"
                )
        else:  # NO_RECOMMENDATION
            if self.no_recommendation_reason is None:
                raise ValueError(
                    "no_recommendation_reason is required when status='no_recommendation'"
                )
            if self.calibration_token is not None:
                raise ValueError(
                    "calibration_token must be None when status='no_recommendation'"
                )
        return self


# ── ApplyCalibrationRequest / ApplyCalibrationResult ──────────────────────────

class TaskPendingRebind(BaseModel):
    """
    A task still bound to the superseded condition version.

    Returned inside ApplyCalibrationResult.tasks_pending_rebind.
    Informational only — callers must rebind explicitly via PATCH /tasks/{id}.
    """
    task_id: str
    intent: str


class ApplyCalibrationRequest(BaseModel):
    """
    Request body for POST /conditions/apply-calibration.

    calibration_token is the only path to applying calibrated params.
    Do NOT add condition_id, condition_version, threshold, or explicit params.
    The token resolves all of these server-side, ensuring atomicity.

    new_version is optional. When absent, the service auto-increments the
    source version (e.g. '1.0' → '1.1').
    """
    calibration_token: str
    new_version: str | None = None


class ApplyCalibrationResult(BaseModel):
    """
    Output of POST /conditions/apply-calibration.

    A new immutable condition version has been registered with the
    recommended parameters from the token.

    tasks_pending_rebind lists tasks still bound to previous_version.
    This is INFORMATIONAL ONLY — the service never auto-rebinds tasks.
    Callers must rebind explicitly via PATCH /tasks/{id} with new_version.
    """
    condition_id: str
    previous_version: str
    new_version: str
    params_applied: dict[str, Any]
    tasks_pending_rebind: list[TaskPendingRebind] = Field(default_factory=list)


# ── Calibration telemetry (opt-in platform analytics) ─────────────────────────

class CalibrationTelemetry(BaseModel):
    """
    Anonymised structural telemetry emitted after a successful calibration cycle.

    Opt-in only. Never contains customer data, entity values, or decision outcomes.
    The CalibrationService must emit this after each successful cycle when the
    tenant has opted in.

    tenant_id_hash is a one-way hash — it cannot be reversed to the original ID.
    """
    tenant_id_hash: str
    strategy_type: str
    domain_tag: str
    initial_params: dict[str, Any]
    stable_params: dict[str, Any]
    calibration_cycles_to_stable: int
    alerts_per_entity_per_month: float
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
