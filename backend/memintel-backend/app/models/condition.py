"""
app/models/condition.py
──────────────────────────────────────────────────────────────────────────────
Condition domain models.

Covers every layer of the φ (phi) layer:
  - Strategy type classification and per-strategy parameter models
  - StrategyDefinition — a discriminated union keyed on the strategy type
  - ConditionDefinition — a registered, versioned condition
  - DecisionValue — runtime output of a strategy evaluation (carries provenance)
  - Explain models — ConditionExplanation, DecisionExplanation

Design notes
────────────
Strategy params are typed models, not raw dicts.
  Each strategy has a separate params class (ThresholdParams, ZScoreParams, …)
  with exact field names as specified. This makes naming bugs compile-time
  errors: z_score uses `threshold`, not `value`; percentile uses `top`/`bottom`,
  not `above`/`below`. The strategy implementations in app/strategies/ receive
  the typed params model and never parse a raw dict.

StrategyDefinition is a discriminated union.
  The `type` field is the discriminator. Pydantic v2 routes construction and
  validation to the correct variant automatically, so the caller never needs to
  branch on type before accessing params fields. The wire format is unchanged —
  it serialises to {"type": "threshold", "params": {"direction": "above", ...}}.

BOOLEAN_STRATEGIES / CATEGORICAL_STRATEGIES constants are defined here.
  The compiler imports these to determine the output decision type. They are
  the single source of truth — never duplicated in the strategy implementations.

TYPE_STRATEGY_COMPATIBILITY maps each primitive type to its valid strategies.
  Derived directly from memintel.guardrails.md §4. The compiler imports this
  for type-checking condition definitions at compile time.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.task import Namespace


# ── Enums ─────────────────────────────────────────────────────────────────────

class StrategyType(str, Enum):
    """
    Canonical strategy type identifiers. String values are the wire values.

    Critical spellings (from py-instructions.md — these are non-obvious):
      Z_SCORE  = 'z_score'   NOT 'zscore' — used everywhere: API, YAML, guardrails
    """
    THRESHOLD  = "threshold"
    PERCENTILE = "percentile"
    Z_SCORE    = "z_score"
    CHANGE     = "change"
    EQUALS     = "equals"
    COMPOSITE  = "composite"


class DecisionType(str, Enum):
    """
    Output type of a condition evaluation.

    boolean     — produced by threshold, percentile, z_score, change, composite
    categorical — produced by equals only
    """
    BOOLEAN     = "boolean"
    CATEGORICAL = "categorical"


# ── Strategy direction enums ──────────────────────────────────────────────────
# Separate enums per strategy because the valid direction values differ.
# Using Literal types inside params models is more precise, but enums here
# are useful for validation in the guardrails store and LLM prompt injection.

class ThresholdDirection(str, Enum):
    ABOVE = "above"
    BELOW = "below"


class PercentileDirection(str, Enum):
    """top / bottom — NOT above / below (a common source of bugs)."""
    TOP    = "top"
    BOTTOM = "bottom"


class ZScoreDirection(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    ANY   = "any"


class ChangeDirection(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    ANY      = "any"


class CompositeOperator(str, Enum):
    AND = "AND"
    OR  = "OR"


# ── Per-strategy parameter models ─────────────────────────────────────────────
# Each model enforces the exact parameter names and constraints from the
# guardrails spec (§3) and py-instructions.md strategy descriptions.

class ThresholdParams(BaseModel):
    """
    threshold strategy parameters.

    direction: 'above' | 'below'
    value:     the fixed cutoff (any float — no global bounds)

    Note: parameter key is 'value', NOT 'cutoff'.
    Fires when: direction='above' → concept_value > value
                direction='below' → concept_value < value
    """
    direction: ThresholdDirection
    value: float


class PercentileParams(BaseModel):
    """
    percentile strategy parameters.

    direction: 'top' | 'bottom'  (NOT 'above'/'below')
    value:     percentile cutoff in range [0, 100]
    """
    direction: PercentileDirection
    value: float = Field(ge=0.0, le=100.0)


class ZScoreParams(BaseModel):
    """
    z_score strategy parameters.

    threshold: number of standard deviations required to trigger (must be > 0)
    direction: 'above' | 'below' | 'any'
    window:    lookback duration string (default '30d')

    Note: parameter key is 'threshold', NOT 'value' or 'cutoff'.
    This is different from every other strategy — z_score is the only one
    where the sensitivity param is called 'threshold'.
    """
    threshold: float = Field(gt=0.0)
    direction: ZScoreDirection
    window: str = "30d"


class ChangeParams(BaseModel):
    """
    change strategy parameters.

    direction: 'increase' | 'decrease' | 'any'
    value:     percentage change threshold (must be >= 0)
    window:    lookback duration string (default '1d')

    Note: parameter key is 'value' (not 'threshold' or 'cutoff').
    """
    direction: ChangeDirection
    value: float = Field(ge=0.0)
    window: str = "1d"


class EqualsParams(BaseModel):
    """
    equals strategy parameters.

    value:  the label to match
    labels: optional closed set of valid labels (subset of the categorical
            type's declared label set). When provided, 'value' must be a
            member of this list. Validated at compile time against the
            concept's declared categorical labels.

    Calibration note: equals has no numeric parameter to adjust.
    POST /conditions/calibrate always returns no_recommendation with
    reason='not_applicable_strategy' for conditions using this strategy.
    """
    value: str
    labels: list[str] | None = None

    @field_validator("labels")
    @classmethod
    def _labels_not_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("labels must be null or a non-empty list")
        return v


class CompositeParams(BaseModel):
    """
    composite strategy parameters.

    operator: 'AND' | 'OR'
    operands: list of condition_ids to evaluate (minimum 2)

    Compiler enforcement rules (checked at compile time, not here):
      - all operands must produce decision<boolean> — equals is excluded
      - composite cannot be nested inside another composite's operands
      - all operand condition_ids must exist in the registry

    Minimum 2 operands is enforced here at model construction time.
    """
    operator: CompositeOperator
    operands: list[str] = Field(min_length=2)


# ── Discriminated StrategyDefinition union ────────────────────────────────────
# Each variant has a Literal `type` field — Pydantic v2 uses this as the
# discriminator to select the correct params model during construction.
#
# StrategyDefinition is the type used in ConditionDefinition.strategy.
# Wire format: {"type": "threshold", "params": {"direction": "above", "value": 0.8}}

class ThresholdStrategy(BaseModel):
    type: Literal[StrategyType.THRESHOLD]
    params: ThresholdParams


class PercentileStrategy(BaseModel):
    type: Literal[StrategyType.PERCENTILE]
    params: PercentileParams


class ZScoreStrategy(BaseModel):
    type: Literal[StrategyType.Z_SCORE]
    params: ZScoreParams


class ChangeStrategy(BaseModel):
    type: Literal[StrategyType.CHANGE]
    params: ChangeParams


class EqualsStrategy(BaseModel):
    type: Literal[StrategyType.EQUALS]
    params: EqualsParams


class CompositeStrategy(BaseModel):
    type: Literal[StrategyType.COMPOSITE]
    params: CompositeParams


#: Discriminated union — the single type used for ConditionDefinition.strategy.
#: Pydantic v2 resolves the correct variant from the 'type' field at parse time.
StrategyDefinition = Annotated[
    ThresholdStrategy
    | PercentileStrategy
    | ZScoreStrategy
    | ChangeStrategy
    | EqualsStrategy
    | CompositeStrategy,
    Field(discriminator="type"),
]


# ── Strategy classification constants ─────────────────────────────────────────
# Used by the compiler to determine output decision type without strategy
# logic. Single source of truth — do not duplicate in strategy implementations.

#: Strategies whose evaluate() returns decision<boolean>.
BOOLEAN_STRATEGIES: frozenset[StrategyType] = frozenset({
    StrategyType.THRESHOLD,
    StrategyType.PERCENTILE,
    StrategyType.Z_SCORE,
    StrategyType.CHANGE,
    StrategyType.COMPOSITE,
})

#: Strategies whose evaluate() returns decision<categorical>.
CATEGORICAL_STRATEGIES: frozenset[StrategyType] = frozenset({
    StrategyType.EQUALS,
})

assert BOOLEAN_STRATEGIES | CATEGORICAL_STRATEGIES == frozenset(StrategyType), (
    "Strategy classification is incomplete — every StrategyType must be in "
    "exactly one of BOOLEAN_STRATEGIES or CATEGORICAL_STRATEGIES."
)
assert BOOLEAN_STRATEGIES.isdisjoint(CATEGORICAL_STRATEGIES), (
    "A strategy cannot belong to both BOOLEAN_STRATEGIES and CATEGORICAL_STRATEGIES."
)


# ── Type-strategy compatibility map ───────────────────────────────────────────
# Source: memintel.guardrails.md §4.
# Used by the compiler to validate condition inputs against primitive types.
# Key: primitive type string. Value: set of valid strategies for that type.

TYPE_STRATEGY_COMPATIBILITY: dict[str, frozenset[StrategyType]] = {
    "float":             frozenset({StrategyType.THRESHOLD, StrategyType.PERCENTILE}),
    "int":               frozenset({StrategyType.THRESHOLD, StrategyType.PERCENTILE}),
    "time_series<float>": frozenset({
        StrategyType.CHANGE, StrategyType.Z_SCORE,
        StrategyType.PERCENTILE, StrategyType.THRESHOLD,
    }),
    "time_series<int>":  frozenset({
        StrategyType.CHANGE, StrategyType.Z_SCORE,
        StrategyType.PERCENTILE, StrategyType.THRESHOLD,
    }),
    "categorical":       frozenset({StrategyType.EQUALS}),
    "string":            frozenset({StrategyType.EQUALS}),
    # boolean primitives: no valid condition strategies.
    # Use a concept to derive a float or categorical signal first.
    "boolean":           frozenset(),
}


# ── ConditionDefinition ───────────────────────────────────────────────────────

class ConditionDefinition(BaseModel):
    """
    A registered, versioned condition definition.

    Stored in the `definitions` table with definition_type='condition'.
    The body JSONB column holds the serialisation of this model.

    Immutability: once registered, a (condition_id, version) pair is permanent.
    Updates require a new version. The unique constraint on the definitions
    table enforces this at the DB level.

    Deprecation: deprecated conditions are retained for audit. All existing
    task references continue to resolve. deprecated=True is advisory — the
    runtime does not automatically reject deprecated conditions, but the
    rebinding validation step warns when a task is being rebound to a
    deprecated condition version.
    """
    condition_id: str
    version: str
    concept_id: str
    concept_version: str
    strategy: StrategyDefinition
    namespace: Namespace
    created_at: datetime | None = None
    deprecated: bool = False


# ── DecisionValue ─────────────────────────────────────────────────────────────

class DecisionValue(BaseModel):
    """
    Runtime output of a strategy evaluation. Returned by all six strategies.

    Carries full provenance — which condition and version produced the decision,
    for which entity, at what timestamp. This provenance is consumed by:
      - ActionTrigger (routes the decision to the correct actions)
      - FeedbackStore (links feedback to a specific decision via timestamp)
      - ExplanationService (reconstructs the evaluation context)

    value is bool for BOOLEAN_STRATEGIES, str (matched label) for EQUALS.

    DecisionValue is NOT assignable to bool or str without unwrapping.
    Actions receive DecisionValue directly — they do not unwrap it.
    Use unwrap() to extract .value and discard provenance.
    """
    value: bool | str
    decision_type: DecisionType
    condition_id: str
    condition_version: str
    entity: str
    timestamp: str | None = None

    def unwrap(self) -> bool | str:
        """Extract the raw value, discarding provenance."""
        return self.value


# ── Explainability models ─────────────────────────────────────────────────────

class ConditionExplanation(BaseModel):
    """
    Explains the logic and parameters of a condition definition.

    Returned by POST /conditions/explain.
    Deterministic given (condition_id, condition_version) — application
    context may influence wording but MUST NOT influence strategy type,
    parameter values, or the natural language summary's factual claims.
    """
    condition_id: str
    condition_version: str
    strategy: StrategyDefinition
    concept_id: str
    concept_version: str
    natural_language_summary: str
    parameter_rationale: str


class DriverContribution(BaseModel):
    """
    Attribution of a single input signal to a concept value in a decision.

    Used inside DecisionExplanation.drivers to break down how each primitive
    contributed to the concept output that was evaluated by the condition.
    """
    signal: str
    contribution: float
    value: float | int | bool | str


class DecisionExplanation(BaseModel):
    """
    Explains a specific decision result for an entity at a given timestamp.

    Returned by POST /decisions/explain.
    Fully deterministic given (condition_id, condition_version, entity,
    timestamp). Application context has zero influence on decision, parameter
    values, or attribution weights (see guardrails.md §2.5).

    Naming alignment (from API spec x-ts-note):
      DecisionResult.value     ← the decision output in evaluation results
      DecisionExplanation.decision ← the same concept in explain responses
      These use different field names by spec design — do not rename either.

    threshold_applied: present for threshold/percentile/change/z_score.
      Null for equals (no numeric parameter) and composite (operator, not threshold).
    label_matched: present for equals only.
      Null for all boolean strategies.
    """
    condition_id: str
    condition_version: str
    entity: str
    timestamp: datetime | None
    decision: bool | str
    decision_type: DecisionType
    concept_value: float | int | str
    strategy_type: StrategyType
    threshold_applied: float | None = None
    label_matched: str | None = None
    drivers: list[DriverContribution] = Field(default_factory=list)
