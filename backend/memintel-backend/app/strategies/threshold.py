"""
app/strategies/threshold.py
────────────────────────────────────────────────────────────────────────────────
Threshold condition strategy.

Spec (py-instructions.md):
  params: { direction: 'above' | 'below', value: float }
  Logic:  direction='above' → fires when result.value > params['value']
          direction='below' → fires when result.value < params['value']
  Input:  float or int
  Output: decision<boolean>

Note: the parameter key is 'value', NOT 'cutoff'.
"""
from __future__ import annotations

from app.models.condition import DecisionValue
from app.models.result import ConceptResult
from app.strategies.base import (
    ConditionStrategy,
    require_numeric,
    require_param,
    semantic_error,
)

_STRATEGY = "threshold"
_VALID_DIRECTIONS = frozenset({"above", "below"})


class ThresholdStrategy(ConditionStrategy):
    """
    Fires when the concept value crosses a fixed numeric cutoff.

    Use for: churn > 0.8, latency > 500 ms, days_inactive > 14.
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
        Evaluate threshold condition.

        Raises
        ------
        MemintelError(type_error)     if result.type is not float or int.
        MemintelError(semantic_error) if direction or value params are missing
                                      or direction is not 'above' / 'below'.
        """
        require_numeric(result, _STRATEGY)

        direction = require_param(params, "direction", _STRATEGY)
        threshold = require_param(params, "value", _STRATEGY)

        if direction not in _VALID_DIRECTIONS:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['direction'] must be 'above' or 'below', "
                f"got '{direction}'.",
                location="params.direction",
            )

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['value'] must be numeric, "
                f"got '{threshold}'.",
                location="params.value",
            )

        current = float(result.value)

        if direction == "above":
            fired = current > threshold
        else:  # below
            fired = current < threshold

        return self._boolean_decision(fired, result, condition_id, condition_version)
