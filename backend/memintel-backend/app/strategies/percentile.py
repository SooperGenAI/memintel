"""
app/strategies/percentile.py
────────────────────────────────────────────────────────────────────────────────
Percentile condition strategy.

Spec (py-instructions.md):
  params: { direction: 'top' | 'bottom', value: float (0-100) }
  Logic:  rank result.value against history; compare rank to params['value'].
          direction='top'    → fires if value is in the top X% of history
          direction='bottom' → fires if value is in the bottom X% of history
  Input:  float or int
  Output: decision<boolean>

Note: direction values are 'top'/'bottom', NOT 'above'/'below'.

Rank computation:
  'top' X%    → fires when current_value > percentile(100 - X) of history
  'bottom' X% → fires when current_value < percentile(X) of history

If history is empty, the condition cannot be evaluated and does not fire.
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

_STRATEGY = "percentile"
_VALID_DIRECTIONS = frozenset({"top", "bottom"})


def _percentile(values: list[float], p: float) -> float:
    """
    Compute the p-th percentile (0-100) of ``values`` using linear interpolation.
    ``values`` must be non-empty and already sorted.
    """
    n = len(values)
    if n == 1:
        return values[0]
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return values[lower] * (1.0 - frac) + values[upper] * frac


class PercentileStrategy(ConditionStrategy):
    """
    Fires when the concept value ranks in the top or bottom X% of history.

    Use for: top 10% LTV, bottom 25th percentile retention.
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
        Evaluate percentile condition.

        Raises
        ------
        MemintelError(type_error)     if result.type is not float or int.
        MemintelError(semantic_error) if direction or value params are missing,
                                      direction is not 'top'/'bottom', or value
                                      is outside [0, 100].
        """
        require_numeric(result, _STRATEGY)

        direction = require_param(params, "direction", _STRATEGY)
        cutoff    = require_param(params, "value", _STRATEGY)

        if direction not in _VALID_DIRECTIONS:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['direction'] must be 'top' or 'bottom', "
                f"got '{direction}'.",
                location="params.direction",
            )

        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['value'] must be numeric (0-100), "
                f"got '{cutoff}'.",
                location="params.value",
            )

        if not (0.0 <= cutoff <= 100.0):
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['value'] must be in [0, 100], "
                f"got {cutoff}.",
                location="params.value",
            )

        if not history:
            return self._boolean_decision(False, result, condition_id, condition_version)

        sorted_history = sorted(float(h.value) for h in history)
        current = float(result.value)

        if direction == "top":
            threshold_pct = _percentile(sorted_history, 100.0 - cutoff)
            fired = current > threshold_pct
        else:  # bottom
            threshold_pct = _percentile(sorted_history, cutoff)
            fired = current < threshold_pct

        return self._boolean_decision(fired, result, condition_id, condition_version)
