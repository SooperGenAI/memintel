"""
app/strategies/z_score.py
────────────────────────────────────────────────────────────────────────────────
Z-Score condition strategy.

Spec (py-instructions.md):
  params: { threshold: float, direction: 'above' | 'below' | 'any', window: duration }
  Logic:  z = (value - mean(history)) / std(history)
          direction='above' → fires if z  >  params['threshold']
          direction='below' → fires if z  < -params['threshold']
          direction='any'   → fires if |z| > params['threshold']
  Input:  float or int
  Output: decision<boolean>

Note: the parameter key is 'threshold', NOT 'value' or 'cutoff'.
      This is the only strategy where the sensitivity param is named 'threshold'.

Edge cases:
  - history is empty     → cannot compute z-score; does not fire.
  - std(history) == 0    → all historical values are identical; z is undefined;
                           does not fire (the value has not deviated from baseline).
"""
from __future__ import annotations

import math

from app.models.condition import DecisionValue
from app.models.result import ConceptResult
from app.strategies.base import (
    ConditionStrategy,
    require_numeric,
    require_param,
    semantic_error,
)

_STRATEGY = "z_score"
_VALID_DIRECTIONS = frozenset({"above", "below", "any"})


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mean: float) -> float:
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


class ZScoreStrategy(ConditionStrategy):
    """
    Fires when the concept value deviates from the rolling mean by more than
    ``threshold`` standard deviations.

    Use for: DAU anomaly detection, payment failure spike, volatility spike.
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
        Evaluate z_score condition.

        Raises
        ------
        MemintelError(type_error)     if result.type is not float or int.
        MemintelError(semantic_error) if threshold or direction params are missing,
                                      or direction is not 'above'/'below'/'any',
                                      or threshold is not > 0.
        """
        require_numeric(result, _STRATEGY)

        threshold = require_param(params, "threshold", _STRATEGY)
        direction = require_param(params, "direction", _STRATEGY)

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['threshold'] must be numeric, "
                f"got '{threshold}'.",
                location="params.threshold",
            )

        if threshold <= 0:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['threshold'] must be > 0, "
                f"got {threshold}.",
                location="params.threshold",
            )

        if direction not in _VALID_DIRECTIONS:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['direction'] must be "
                f"'above', 'below', or 'any', got '{direction}'.",
                location="params.direction",
            )

        if not history:
            return self._boolean_decision(False, result, condition_id, condition_version)

        history_vals = [float(h.value) for h in history]
        mean = _mean(history_vals)
        std  = _std(history_vals, mean)

        if std == 0.0:
            # No deviation possible — condition does not fire.
            return self._boolean_decision(False, result, condition_id, condition_version)

        z = (float(result.value) - mean) / std

        if direction == "above":
            fired = z > threshold
        elif direction == "below":
            fired = z < -threshold
        else:  # any
            fired = abs(z) > threshold

        return self._boolean_decision(fired, result, condition_id, condition_version)
