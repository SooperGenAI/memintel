"""
app/strategies/composite.py
────────────────────────────────────────────────────────────────────────────────
Composite condition strategy.

Spec (py-instructions.md):
  params: { operator: 'AND' | 'OR' | 'NOT', operands: list[condition_id],
            operand_results: list[DecisionValue] }
  Logic:  1. Evaluate each operand condition independently (caller's responsibility).
          2. Apply operator (AND | OR | NOT) across the boolean results.
          3. Return a single decision<boolean>.
  Input:  decision<boolean> ONLY — composite cannot wrap equals conditions.
  Output: decision<boolean>

Runtime contract:
  The composite strategy does NOT evaluate sub-conditions itself.  The caller
  (ConditionEvaluator) must evaluate each operand condition first, then pass the
  results as params['operand_results'] (list[DecisionValue]).

  This keeps the strategy stateless and testable in isolation.

Compiler enforcement (checked at compile time by Validator, not here):
  - operands must all produce decision<boolean>
  - equals conditions produce decision<categorical> → type_error if used as operand
  - composite cannot be nested inside another composite's operands → semantic_error
  - operands list must contain at least 2 condition_ids for AND/OR → semantic_error if fewer
  - all operand condition_ids must exist in the registry → reference_error if not

Runtime enforcement (checked here):
  - params['operator'] must be 'AND', 'OR', or 'NOT' → semantic_error
  - params['operand_results'] must be provided → semantic_error
  - NOT requires exactly one operand → semantic_error if count != 1
  - NOT cannot be applied to decision<categorical> → type_error
  - NOT propagates reason from operand (null_input, insufficient_history, etc.)
    without inverting — the operand result is unevaluated or qualified
  - each operand DecisionValue for AND/OR must have decision_type=BOOLEAN → type_error
    (decision<categorical> from equals strategy is rejected)
"""
from __future__ import annotations

from app.models.condition import DecisionType, DecisionValue
from app.models.result import ConceptResult
from app.strategies.base import (
    ConditionStrategy,
    require_param,
    semantic_error,
    type_error,
)

_STRATEGY = "composite"
_VALID_OPERATORS = frozenset({"AND", "OR", "NOT"})


class CompositeStrategy(ConditionStrategy):
    """
    Combines two or more decision<boolean> results using AND or OR logic.

    Use for: multi-factor decisions such as
    (high_churn AND high_value) OR critical_event.
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
        Evaluate composite condition.

        Parameters
        ----------
        result:            Not used by this strategy (composite combines decisions,
                           not concept values).
        history:           Not used by this strategy.
        params:            Must contain:
                             operator:        'AND' | 'OR'
                             operand_results: list[DecisionValue] — pre-evaluated
                                             operand decisions from the caller.

        Raises
        ------
        MemintelError(semantic_error) if operator is missing or not 'AND'/'OR'.
        MemintelError(semantic_error) if operand_results is missing or empty.
        MemintelError(type_error)     if any operand has decision_type != BOOLEAN
                                      (e.g. decision<categorical> from equals).
        """
        operator        = require_param(params, "operator", _STRATEGY)
        operand_results = require_param(params, "operand_results", _STRATEGY)

        if operator not in _VALID_OPERATORS:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['operator'] must be 'AND', 'OR', or 'NOT', "
                f"got '{operator}'.",
                location="params.operator",
            )

        if not isinstance(operand_results, list) or len(operand_results) == 0:
            raise semantic_error(
                f"Strategy '{_STRATEGY}': params['operand_results'] must be a "
                f"non-empty list of DecisionValue objects.",
                location="params.operand_results",
            )

        if operator == "NOT":
            # RULE 1: NOT requires exactly one operand
            if len(operand_results) != 1:
                raise semantic_error(
                    f"Strategy '{_STRATEGY}': NOT operator requires exactly one operand, "
                    f"got {len(operand_results)}.",
                    location="params.operand_results",
                )
            operand = operand_results[0]
            if not isinstance(operand, DecisionValue):
                raise semantic_error(
                    f"Strategy '{_STRATEGY}': operand_results[0] must be a "
                    f"DecisionValue, got '{type(operand).__name__}'.",
                    location="params.operand_results[0]",
                )
            # RULE 4: any reason (null_input, insufficient_history, zero_variance,
            # infinite_change, threshold_zero, …) means the operand result is
            # unevaluated or qualified — propagate the reason and return False without
            # inverting.  Inversion on an unevaluated result is meaningless.
            if operand.reason is not None:
                return self._boolean_decision(
                    False, result, condition_id, condition_version,
                    reason=operand.reason,
                    history_count=operand.history_count,
                )
            # RULE 2: categorical result cannot be logically inverted
            if operand.decision_type == DecisionType.CATEGORICAL:
                raise type_error(
                    f"Strategy '{_STRATEGY}': NOT operator cannot be applied to a "
                    f"categorical result — use equals strategy with a specific target "
                    f"label instead.",
                    location="params.operand_results[0].decision_type",
                )
            # RULE 3: invert boolean result
            return self._boolean_decision(
                not bool(operand.value), result, condition_id, condition_version,
                reason=None,
                history_count=operand.history_count,
            )

        # Validate each operand is decision<boolean> (AND / OR path)
        for i, operand in enumerate(operand_results):
            if not isinstance(operand, DecisionValue):
                raise semantic_error(
                    f"Strategy '{_STRATEGY}': operand_results[{i}] must be a "
                    f"DecisionValue, got '{type(operand).__name__}'.",
                    location=f"params.operand_results[{i}]",
                )
            if operand.decision_type != DecisionType.BOOLEAN:
                raise type_error(
                    f"Strategy '{_STRATEGY}': operand_results[{i}] has "
                    f"decision_type='{operand.decision_type.value}' but composite "
                    f"requires decision<boolean>.  "
                    f"Operand condition '{operand.condition_id}' likely uses the "
                    f"'equals' strategy which produces decision<categorical>.",
                    location=f"params.operand_results[{i}].decision_type",
                )

        # Apply logical operator
        bool_values = [bool(op.value) for op in operand_results]

        if operator == "AND":
            fired = all(bool_values)
        else:  # OR
            fired = any(bool_values)

        # The composite result entity/timestamp come from the caller's context;
        # use result.entity and result.timestamp as the provenance anchor.
        return self._boolean_decision(fired, result, condition_id, condition_version)
