"""
app/strategies/change.py
────────────────────────────────────────────────────────────────────────────────
Change condition strategy.

Spec (py-instructions.md):
  params: { direction: 'increase' | 'decrease' | 'any', value: float, window: duration }
  Logic:  pct_change = (current - previous) / abs(previous)
          direction='increase' → fires if pct_change  >  params['value']
          direction='decrease' → fires if pct_change  < -params['value']
          direction='any'      → fires if |pct_change| > params['value']
  Input:  float or int
  Output: decision<boolean>

Note: direction values are 'increase'/'decrease'/'any'.
      The parameter key is 'value' (not 'threshold' or 'cutoff').
      params['value'] is a decimal fraction (0.1 = 10% change).

"previous" is the most recent historical result (history[-1]).

Edge cases:
  - history is empty          → cannot compute change; does not fire.
  - previous value is 0       → division by zero; does not fire.
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

_STRATEGY = "change"
_VALID_DIRECTIONS = frozenset({"increase", "decrease", "any"})


class ChangeStrategy(ConditionStrategy):
    """
    Fires when the concept value has changed by more than X relative to the
    most recent historical value.

    Use for: stock price movement, engagement drop, metric spike.
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
        Evaluate change condition.

        Raises
        ------
        MemintelError(type_error)     if result.type is not float or int.
        MemintelError(semantic_error) if direction or value params are missing,
                                      direction is not 'increase'/'decrease'/'any',
                                      or value is not >= 0.
        """
        require_numeric(result, _STRATEGY)

        direction = require_param(params, "direction", _STRATEGY)
        threshold = require_param(params, "value", _STRATEGY)

        if direction not in _VALID_DIRECTIONS:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['direction'] must be "
                f"'increase', 'decrease', or 'any', got '{direction}'.",
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

        if threshold < 0:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['value'] must be >= 0, "
                f"got {threshold}.",
                location="params.value",
            )

        if not history:
            return self._boolean_decision(False, result, condition_id, condition_version)

        previous = float(history[-1].value)
        if previous == 0.0:
            # Cannot compute percentage change — does not fire.
            return self._boolean_decision(False, result, condition_id, condition_version)

        current    = float(result.value)
        pct_change = (current - previous) / abs(previous)

        if direction == "increase":
            fired = pct_change > threshold
        elif direction == "decrease":
            fired = pct_change < -threshold
        else:  # any
            fired = abs(pct_change) > threshold

        return self._boolean_decision(fired, result, condition_id, condition_version)
