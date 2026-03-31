"""
app/strategies/base.py
────────────────────────────────────────────────────────────────────────────────
Abstract base class for all six Memintel condition strategies.

Contract (py-instructions.md "Condition Strategies"):
  evaluate(result, history, params, *, condition_id, condition_version)
      → DecisionValue          — always; never a raw bool or str
      Raises type_error        — if result.type is invalid for this strategy
      Raises semantic_error    — if required params are missing or wrong type

Every concrete strategy must:
  1. Inherit from ConditionStrategy.
  2. Implement evaluate().
  3. Return DecisionValue with correct decision_type (boolean or categorical).
  4. Raise MemintelError(type_error) for invalid input types.
  5. Raise MemintelError(semantic_error) for missing / malformed params.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.condition import DecisionType, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import ConceptResult


# ── Error factories ─────────────────────────────────────────────────────────────

def type_error(message: str, location: str | None = None) -> MemintelError:
    """Return a MemintelError with error_type=TYPE_ERROR."""
    return MemintelError(ErrorType.TYPE_ERROR, message, location=location)


def semantic_error(message: str, location: str | None = None) -> MemintelError:
    """Return a MemintelError with error_type=SEMANTIC_ERROR."""
    return MemintelError(ErrorType.SEMANTIC_ERROR, message, location=location)


# ── Type-check helpers ─────────────────────────────────────────────────────────

#: Result types accepted by numeric strategies (threshold, percentile, z_score, change).
_NUMERIC_TYPES: frozenset[str] = frozenset({"float", "int"})

#: Result types accepted by the equals strategy.
_TEXT_TYPES: frozenset[str] = frozenset({"categorical", "string"})


def require_numeric(result: ConceptResult, strategy: str) -> None:
    """Raise type_error if result.type is not numeric (float or int)."""
    if result.type.value not in _NUMERIC_TYPES:
        raise type_error(
            f"Strategy '{strategy}' requires a numeric input (float or int), "
            f"but received type '{result.type.value}'.",
            location="result.type",
        )


def require_text(result: ConceptResult, strategy: str) -> None:
    """Raise type_error if result.type is not categorical or string."""
    if result.type.value not in _TEXT_TYPES:
        raise type_error(
            f"Strategy '{strategy}' requires a categorical or string input, "
            f"but received type '{result.type.value}'.",
            location="result.type",
        )


def require_param(params: dict, key: str, strategy: str) -> object:
    """Raise semantic_error if ``key`` is missing from params. Returns the value."""
    if key not in params or params[key] is None:
        raise semantic_error(
            f"Strategy '{strategy}' requires params['{key}'] but it was not provided.",
            location=f"params.{key}",
        )
    return params[key]


# ── Base class ──────────────────────────────────────────────────────────────────

class ConditionStrategy(ABC):
    """
    Abstract base class for all Memintel condition strategies.

    Subclasses implement evaluate() which takes the current concept result,
    historical results for the reference frame, and the strategy parameters,
    and returns a DecisionValue carrying the decision and its provenance.

    evaluate() MUST:
      - Return DecisionValue — never a raw bool or str.
      - Set decision_type to DecisionType.BOOLEAN or DecisionType.CATEGORICAL.
      - Set condition_id, condition_version, entity from the call-site arguments.
      - Raise MemintelError(type_error) for invalid input types.
      - Raise MemintelError(semantic_error) for invalid / missing params.
    """

    @abstractmethod
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
        Evaluate the condition and return a DecisionValue.

        Parameters
        ----------
        result:            Current concept result (Rₜ).
        history:           Past concept results for the same entity (reference frame).
                           Order: oldest first, most recent last.
        params:            Strategy parameters (direction, value, threshold, …).
        condition_id:      Provenance — condition definition id.
        condition_version: Provenance — condition version string.
        """

    # ── Shared helper for building DecisionValues ───────────────────────────────

    @staticmethod
    def _boolean_decision(
        fired: bool,
        result: ConceptResult,
        condition_id: str,
        condition_version: str,
        reason: str | None = None,
        history_count: int | None = None,
    ) -> DecisionValue:
        """Build a decision<boolean> DecisionValue from an evaluation result."""
        return DecisionValue(
            value=fired,
            decision_type=DecisionType.BOOLEAN,
            condition_id=condition_id,
            condition_version=condition_version,
            entity=result.entity,
            timestamp=result.timestamp,
            reason=reason,
            history_count=history_count,
        )

    @staticmethod
    def _categorical_decision(
        matched_label: str,
        result: ConceptResult,
        condition_id: str,
        condition_version: str,
    ) -> DecisionValue:
        """Build a decision<categorical> DecisionValue from an evaluation result."""
        return DecisionValue(
            value=matched_label,
            decision_type=DecisionType.CATEGORICAL,
            condition_id=condition_id,
            condition_version=condition_version,
            entity=result.entity,
            timestamp=result.timestamp,
        )
