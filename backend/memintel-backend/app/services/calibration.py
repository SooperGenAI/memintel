"""
app/services/calibration.py
──────────────────────────────────────────────────────────────────────────────
CalibrationService — condition parameter calibration from feedback.

Implements the full calibration loop:
  calibrate()          → analyse feedback → return CalibrationResult + token
  apply_calibration()  → consume token → register new condition version

Pipeline for calibrate():
  1. Load condition from definition registry
  2. equals / composite strategy → no_recommendation (not_applicable_strategy)
  3. Resolve feedback direction:
       - Explicit feedback_direction overrides aggregation
       - Otherwise: aggregate FeedbackStore records, derive by majority vote
       - No majority or insufficient data → no_recommendation (insufficient_data)
  4. Compute adjusted params via adjust_params() — strategy-aware:
       threshold:  relax → decrease 'value';  tighten → increase 'value'
       percentile: relax → increase 'value';  tighten → decrease 'value'
       change:     relax → decrease 'value';  tighten → increase 'value'
       z_score:    relax → decrease 'threshold'; tighten → increase 'threshold'
     Bounds enforced from guardrails:
       on_bounds_exceeded='clamp'  → clip to bound
       on_bounds_exceeded='reject' → return no_recommendation (bounds_exceeded)
  5. Estimate directional impact
  6. Generate single-use calibration token (24 h expiry — store sets expiry)
  7. Return CalibrationResult

Pipeline for apply_calibration():
  1. Atomically resolve and invalidate the token (None → HTTP 400)
  2. Load source condition from registry
  3. Determine new version (caller-supplied or auto-incremented)
  4. Check version not already registered (ConflictError if duplicate)
  5. Deep-copy source body, set new version + recommended_params
  6. Register new condition version (immutable write)
  7. Find tasks still bound to old version — INFORMATIONAL ONLY
  8. Return ApplyCalibrationResult

Invariants:
  - Never calls the LLM.
  - Never auto-rebinds tasks under any circumstances.
  - The token is consumed exactly once (atomically by CalibrationTokenStore).
  - Existing condition versions are never mutated — only new versions are created.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import structlog

from app.models.calibration import (
    ApplyCalibrationRequest,
    ApplyCalibrationResult,
    CalibrateRequest,
    CalibrationImpact,
    CalibrationResult,
    CalibrationStatus,
    CalibrationToken,
    FeedbackRecord,
    FeedbackValue,
    ImpactDirection,
    MIN_FEEDBACK_THRESHOLD,
    NoRecommendationReason,
    TargetConfig,
    TaskPendingRebind,
)
from app.models.condition import ConditionDefinition, StrategyType
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError

log = structlog.get_logger(__name__)


class CalibrationService:
    """
    Derives parameter recommendations from stored feedback and applies them
    as new immutable condition versions.

    calibrate()         — reads FeedbackStore, computes adjusted params,
                          stores CalibrationToken, returns CalibrationResult.
    apply_calibration() — consumes a CalibrationToken atomically, registers
                          a new ConditionDefinition version, returns
                          ApplyCalibrationResult with tasks_pending_rebind.

    Never calls the LLM. Never auto-rebinds tasks.

    Parameters
    ──────────
    feedback_store      — must implement async get_by_condition(cid, version).
    token_store         — must implement async create(token) and
                          async resolve_and_invalidate(token_string).
    task_store          — must implement async find_by_condition_version(cid, v).
    definition_registry — must implement async get(id, version) and
                          async register(body, namespace, definition_type).
    guardrails_store    — must implement get_guardrails() and
                          get_threshold_bounds(strategy).
    """

    def __init__(
        self,
        feedback_store: Any,
        token_store: Any,
        task_store: Any,
        definition_registry: Any,
        guardrails_store: Any,
        context_store: Any = None,
    ) -> None:
        self._feedback_store    = feedback_store
        self._token_store       = token_store
        self._task_store        = task_store
        self._registry          = definition_registry
        self._guardrails_store  = guardrails_store
        self._context_store     = context_store

    # ── Public API ──────────────────────────────────────────────────────────────

    async def calibrate(self, req: CalibrateRequest) -> CalibrationResult:
        """
        Analyse stored feedback for a condition version and return a
        parameter recommendation + single-use token, or a no_recommendation
        result with a reason.

        Raises NotFoundError if the condition (id, version) is not registered.
        """
        # 1. Load condition
        body = await self._registry.get(req.condition_id, req.condition_version)
        condition = ConditionDefinition.model_validate(body)
        current_params = condition.strategy.params.model_dump()

        # 2. Not-applicable strategies — equals and composite have no numeric
        #    parameter to adjust.
        if condition.strategy.type in (StrategyType.EQUALS, StrategyType.COMPOSITE):
            log.info(
                "calibration_not_applicable",
                condition_id=req.condition_id,
                condition_version=req.condition_version,
                strategy_type=condition.strategy.type.value,
            )
            return CalibrationResult(
                status=CalibrationStatus.NO_RECOMMENDATION,
                no_recommendation_reason=NoRecommendationReason.NOT_APPLICABLE_STRATEGY,
                current_params=current_params,
            )

        # 3. Resolve feedback direction
        if req.feedback_direction:
            direction: str | None = req.feedback_direction
        else:
            records = await self._feedback_store.get_by_condition(
                req.condition_id, req.condition_version
            )
            direction = self.derive_direction(records)

        if direction is None:
            return CalibrationResult(
                status=CalibrationStatus.NO_RECOMMENDATION,
                no_recommendation_reason=NoRecommendationReason.INSUFFICIENT_DATA,
                current_params=current_params,
            )

        # 4. Compute adjusted params
        if self._guardrails_store is None:
            return CalibrationResult(
                status=CalibrationStatus.NO_RECOMMENDATION,
                no_recommendation_reason=NoRecommendationReason.GUARDRAILS_UNAVAILABLE,
                current_params=current_params,
            )

        bounds        = self._guardrails_store.get_threshold_bounds(condition.strategy.type.value)
        guardrails    = self._guardrails_store.get_guardrails()
        on_exceeded   = guardrails.constraints.on_bounds_exceeded

        recommended = self.adjust_params(
            strategy=condition.strategy.type,
            current_params=current_params,
            direction=direction,
            target=req.target,
            bounds=bounds,
            on_bounds_exceeded=on_exceeded,
        )

        if recommended is None:
            return CalibrationResult(
                status=CalibrationStatus.NO_RECOMMENDATION,
                no_recommendation_reason=NoRecommendationReason.BOUNDS_EXCEEDED,
                current_params=current_params,
            )

        # 4b. Apply context calibration bias adjustment.
        #
        # Extract the primary numeric parameter for this strategy:
        #   z_score  → "threshold"   (all others → "value")
        # The statistically_optimal value is the raw output of adjust_params().
        # When an active context with calibration_bias exists, we shift it by
        # a small bias factor and clamp to [0.0, 1.0].
        param_key = "threshold" if condition.strategy.type == StrategyType.Z_SCORE else "value"
        statistically_optimal_val: float | None = None
        context_adjusted_val: float | None = None
        adjustment_explanation: str | None = None

        if param_key in recommended:
            statistically_optimal_val = float(recommended[param_key])

            # Fetch active context — never raises; skip adjustment on any error.
            app_context = None
            if self._context_store is not None:
                try:
                    app_context = await self._context_store.get_active()
                except Exception as exc:
                    log.error("context_fetch_failed_calibration", error=str(exc))

            if app_context is not None and app_context.calibration_bias is not None:
                bias        = app_context.calibration_bias
                bias_dir    = bias.bias_direction
                _BIAS_FACTOR: dict[str, float] = {"high": 0.10, "medium": 0.05, "low": 0.02}

                if bias_dir == "recall":
                    bf           = _BIAS_FACTOR[bias.false_negative_cost]
                    adjusted_val = statistically_optimal_val * (1.0 - bf)
                    cost_label   = f"false_negative_cost={bias.false_negative_cost}"
                elif bias_dir == "precision":
                    bf           = _BIAS_FACTOR[bias.false_positive_cost]
                    adjusted_val = statistically_optimal_val * (1.0 + bf)
                    cost_label   = f"false_positive_cost={bias.false_positive_cost}"
                else:  # balanced — no adjustment
                    adjusted_val = statistically_optimal_val
                    cost_label   = None

                # Clamp to guardrails threshold_bounds, then to [0.0, 1.0] as a
                # safety net.  Bias must never push the value outside the
                # strategy-specific bounds declared in the guardrails.
                raw_adjusted_val = adjusted_val
                bounds_min_f = float(bounds.get("min")) if bounds.get("min") is not None else 0.0
                bounds_max_f = float(bounds.get("max")) if bounds.get("max") is not None else 1.0
                adjusted_val = max(bounds_min_f, min(bounds_max_f, adjusted_val))
                adjusted_val = max(0.0, min(1.0, adjusted_val))

                if adjusted_val != statistically_optimal_val:
                    context_adjusted_val = adjusted_val
                    recommended = dict(recommended)
                    recommended[param_key] = adjusted_val
                    if raw_adjusted_val != adjusted_val:
                        guardrails_bound = (
                            bounds_max_f if raw_adjusted_val > bounds_max_f else bounds_min_f
                        )
                        adjustment_explanation = (
                            f"Threshold adjusted from {statistically_optimal_val:.4f} to "
                            f"{adjusted_val:.4f} toward {bias_dir} based on application "
                            f"context ({cost_label}). Clamped to guardrails bound of "
                            f"{guardrails_bound}."
                        )
                    else:
                        adjustment_explanation = (
                            f"Threshold adjusted from {statistically_optimal_val:.4f} to "
                            f"{adjusted_val:.4f} toward {bias_dir} based on application "
                            f"context ({cost_label})"
                        )

            # Apply meaningful_windows constraint if configured.
            if (
                app_context is not None
                and app_context.behavioural.meaningful_windows is not None
            ):
                mw      = app_context.behavioural.meaningful_windows
                mw_min  = mw.get("min")
                mw_max  = mw.get("max")
                cur_val = float(recommended.get(param_key, statistically_optimal_val))
                clamped = cur_val

                if mw_min is not None and cur_val < float(mw_min):
                    clamped = float(mw_min)
                if mw_max is not None and cur_val > float(mw_max):
                    clamped = float(mw_max)

                if clamped != cur_val:
                    recommended = dict(recommended)
                    recommended[param_key] = clamped
                    window_note = (
                        f"Window parameter clamped to {clamped} per domain context "
                        f"constraints (valid range: {mw_min}-{mw_max})"
                    )
                    if adjustment_explanation:
                        adjustment_explanation += "; " + window_note
                    else:
                        adjustment_explanation = window_note

        # 5. Estimate directional impact
        impact = self._estimate_impact(direction)

        # 6. Generate single-use token (store sets the actual expiry and token_string)
        token_obj = CalibrationToken(
            token_string="pending",                         # overwritten by store
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            recommended_params=recommended,
            expires_at=datetime.now(tz=timezone.utc),       # overwritten by store
        )
        token_string = await self._token_store.create(token_obj)

        log.info(
            "calibration_recommended",
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            strategy_type=condition.strategy.type.value,
            old_params=current_params,
            recommended_params=recommended,
            delta_alerts=impact.delta_alerts,
            feedback_direction=direction,
            statistically_optimal=statistically_optimal_val,
            context_adjusted=context_adjusted_val,
        )

        return CalibrationResult(
            status=CalibrationStatus.RECOMMENDATION_AVAILABLE,
            current_params=current_params,
            recommended_params=recommended,
            calibration_token=token_string,
            impact=impact,
            statistically_optimal=statistically_optimal_val,
            context_adjusted=context_adjusted_val,
            adjustment_explanation=adjustment_explanation,
        )

    async def apply_calibration(
        self, req: ApplyCalibrationRequest
    ) -> ApplyCalibrationResult:
        """
        Consume a calibration token and register a new immutable condition version.

        The calibration_token is the ONLY input path. No condition_id, threshold,
        or explicit params are accepted — the token resolves everything server-side.

        tasks_pending_rebind in the result is INFORMATIONAL ONLY.
        This method NEVER rebinds tasks — callers must do so via PATCH /tasks/{id}.

        Raises:
          MemintelError(PARAMETER_ERROR) — token is invalid, expired, or already used.
          ConflictError                  — new_version already registered.
          NotFoundError                  — source condition not found.
        """
        # 1. Atomically resolve and invalidate token
        token = await self._token_store.resolve_and_invalidate(req.calibration_token)
        if token is None:
            raise MemintelError(
                ErrorType.PARAMETER_ERROR,
                "Invalid or expired calibration token. "
                "Tokens are single-use and expire after 24 hours.",
                location="calibration_token",
                suggestion="Request a new calibration token via POST /conditions/calibrate.",
            )

        condition_id        = token.condition_id
        source_version      = token.condition_version
        recommended_params  = token.recommended_params

        # 2. Load source condition body (raises NotFoundError if missing)
        source_body = await self._registry.get(condition_id, source_version)

        # 3. Determine new version
        new_version = req.new_version or self._auto_increment_version(source_version)

        # 4. Guard: reject if new version is already registered
        try:
            await self._registry.get(condition_id, new_version)
            raise ConflictError(
                f"Condition '{condition_id}' version '{new_version}' already exists. "
                "Choose a different version or omit new_version to auto-increment.",
                location=f"{condition_id}:{new_version}",
            )
        except NotFoundError:
            pass  # Expected — version is available.

        # 5. Build new condition body — never mutate the source
        new_body = copy.deepcopy(source_body)
        new_body["version"] = new_version
        new_body["strategy"]["params"] = recommended_params

        # 6. Register new version (immutable)
        namespace = new_body.get("namespace", "personal")
        if hasattr(namespace, "value"):         # Namespace enum → string
            namespace = namespace.value
        await self._registry.register(
            new_body,
            namespace=namespace,
            definition_type="condition",
        )

        # 7. Find tasks still bound to old version — INFORMATIONAL ONLY.
        #    DO NOT rebind tasks here under any circumstances.
        pending_tasks = await self._task_store.find_by_condition_version(
            condition_id, source_version
        )

        log.info(
            "calibration_applied",
            condition_id=condition_id,
            previous_version=source_version,
            new_version=new_version,
            params_applied=recommended_params,
            tasks_pending_rebind=len(pending_tasks),
        )

        return ApplyCalibrationResult(
            condition_id=condition_id,
            previous_version=source_version,
            new_version=new_version,
            params_applied=recommended_params,
            tasks_pending_rebind=[
                TaskPendingRebind(task_id=t.task_id, intent=t.intent)
                for t in pending_tasks
            ],
        )

    # ── Direction derivation ────────────────────────────────────────────────────

    def derive_direction(self, records: list[FeedbackRecord]) -> str | None:
        """
        Derive calibration direction from a list of feedback records.

        Returns:
          'tighten' — majority false_positive (condition fired when it should not)
          'relax'   — majority false_negative (condition missed when it should fire)
          None      — tie, or fewer than MIN_FEEDBACK_THRESHOLD records (insufficient data)

        correct feedback records are counted but do not shift the majority.
        """
        if len(records) < MIN_FEEDBACK_THRESHOLD:
            return None
        fp_count = sum(1 for r in records if r.feedback == FeedbackValue.FALSE_POSITIVE)
        fn_count = sum(1 for r in records if r.feedback == FeedbackValue.FALSE_NEGATIVE)
        if fp_count > fn_count:
            return "tighten"
        if fn_count > fp_count:
            return "relax"
        return None  # tie — insufficient directional signal

    # ── Parameter adjustment ────────────────────────────────────────────────────

    def adjust_params(
        self,
        strategy: StrategyType,
        current_params: dict[str, Any],
        direction: str,
        target: TargetConfig | None,
        bounds: dict[str, Any],
        on_bounds_exceeded: str,
    ) -> dict[str, Any] | None:
        """
        Compute a strategy-aware parameter adjustment.

        Returns the adjusted params dict, or None when the adjustment would
        violate guardrail bounds and on_bounds_exceeded='reject'.

        Bias semantics (from guardrails §2.4.1):
          threshold:  relax → decrease 'value';  tighten → increase 'value'
          percentile: relax → increase 'value';  tighten → decrease 'value'
          change:     relax → decrease 'value';  tighten → increase 'value'
          z_score:    relax → decrease 'threshold'; tighten → increase 'threshold'

        Step size: 10 % of abs(current_value), minimum 0.1.
        When target (alerts_per_day) is provided the step is not yet target-aware
        (simplified implementation — target support requires historical alert data).
        """
        adjusted  = dict(current_params)
        param_key = "threshold" if strategy == StrategyType.Z_SCORE else "value"
        current_val = float(current_params.get(param_key, 0.0))

        # Step: 10 % of absolute value, minimum 0.1.
        step = max(abs(current_val) * 0.10, 0.1)

        # Signed delta per bias_semantics.
        # percentile inverts the sign relative to all others.
        if strategy == StrategyType.PERCENTILE:
            delta = step if direction == "relax" else -step
        else:
            # threshold, change, z_score: relax → decrease, tighten → increase
            delta = -step if direction == "relax" else step

        new_val   = current_val + delta
        min_bound = bounds.get("min")
        max_bound = bounds.get("max")

        if min_bound is not None and new_val < min_bound:
            if on_bounds_exceeded == "reject":
                return None
            new_val = float(min_bound)

        if max_bound is not None and new_val > max_bound:
            if on_bounds_exceeded == "reject":
                return None
            new_val = float(max_bound)

        adjusted[param_key] = new_val
        return adjusted

    # ── Private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_impact(direction: str) -> CalibrationImpact:
        """
        Return a directional estimate of alert volume change.

        relax   → more alerts  (+1.0, INCREASE)
        tighten → fewer alerts (-1.0, DECREASE)

        The magnitude is a placeholder — real impact estimation requires
        historical alert volume data from the execution log.
        """
        if direction == "relax":
            return CalibrationImpact(delta_alerts=1.0, direction=ImpactDirection.INCREASE)
        return CalibrationImpact(delta_alerts=-1.0, direction=ImpactDirection.DECREASE)

    @staticmethod
    def _auto_increment_version(version: str) -> str:
        """
        Increment the last numeric component of a dot-separated version string.

        Examples:
          '1.0'  → '1.1'
          '1.9'  → '1.10'
          '2.3'  → '2.4'
          '1'    → '2'

        If the last component is not numeric, '.1' is appended.
        """
        parts = version.split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except ValueError:
            parts.append("1")
        return ".".join(parts)
