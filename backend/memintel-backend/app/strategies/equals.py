"""
app/strategies/equals.py
────────────────────────────────────────────────────────────────────────────────
Equals condition strategy.

Spec (py-instructions.md):
  params: { value: string, labels: list[str] | None }
  Logic:  if labels is set, result.value must be in labels (label-set validation).
          fires when result.value == params['value'].
  Input:  categorical or string ONLY.
          Raises type_error on float / int / boolean input.
  Output: decision<categorical>

Calibration note (py-instructions.md):
  POST /conditions/calibrate for equals conditions MUST return
  status='no_recommendation' with reason='not_applicable_strategy'.
  This is enforced in CalibrationService, not here.
"""
from __future__ import annotations

from app.models.condition import DecisionValue
from app.models.result import ConceptResult
from app.strategies.base import (
    ConditionStrategy,
    require_param,
    require_text,
    semantic_error,
    type_error,
)

_STRATEGY = "equals"


class EqualsStrategy(ConditionStrategy):
    """
    Fires when a categorical or string concept value matches a declared label.

    Use for: classification outputs, status-based conditions, segment checks.
    """

    def evaluate(
        self,
        result: ConceptResult,
        history: list[ConceptResult],
        params: dict,
        *,
        condition_id: str = "",
        condition_version: str = "",
    ) -> DecisionValue:
        """
        Evaluate equals condition.

        Raises
        ------
        MemintelError(type_error)     if result.type is not categorical or string.
        MemintelError(semantic_error) if params['value'] is missing.
        MemintelError(type_error)     if params['labels'] is set and result.value
                                      is not in the declared label set.
        """
        if result.value is None:
            return self._boolean_decision(
                False, result, condition_id, condition_version,
                reason="null_input",
                history_count=None,
            )
        require_text(result, _STRATEGY)

        target_label = require_param(params, "value", _STRATEGY)

        if not isinstance(target_label, str):
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['value'] must be a string, "
                f"got '{type(target_label).__name__}'.",
                location="params.value",
            )

        labels = params.get("labels")
        if labels is not None:
            if not isinstance(labels, list) or not labels:
                raise semantic_error(
                    f"Strategy '{_STRATEGY}': params['labels'] must be a non-empty list "
                    f"or null.",
                    location="params.labels",
                )
            current_str = str(result.value)
            if current_str not in labels:
                raise type_error(
                    f"Strategy '{_STRATEGY}': result value '{current_str}' is not in "
                    f"the declared labels {labels}.",
                    location="result.value",
                )

        fired_label = str(result.value) if str(result.value) == target_label else ""

        return self._categorical_decision(
            fired_label, result, condition_id, condition_version
        )
