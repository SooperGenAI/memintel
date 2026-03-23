"""
app/models/guardrails.py
──────────────────────────────────────────────────────────────────────────────
Guardrails domain models — the parsed, validated representation of
`memintel.guardrails.md`.

Source of truth: memintel.guardrails.md v1.4.0

The guardrails file drives three distinct consumers:

  LLM (POST /tasks only)
    strategy_registry, type_compatibility, severity_vocabulary, primitives,
    strategies, thresholds, mappings, priorities, parameter_bias_rules,
    bias_semantics, conflict_resolution, bias_application_rules,
    application_context

  Compiler (all definition registration + validation paths)
    strategy_registry, type_compatibility, constraints
    (application_context and bias rules are EXCLUDED from compiler validation)

  CalibrationService
    constraints.threshold_bounds, constraints.on_bounds_exceeded

Structure overview (mirrors §1 of the spec)
──────────────────────────────────────────
  Guardrails (top-level container)
    ├── application_context     ApplicationContext        §2
    ├── strategy_registry       dict[str, StrategyRegistryEntry]  §3
    ├── type_compatibility      dict[str, TypeCompatibilityEntry] §4
    ├── severity_vocabulary     SeverityVocabulary        §5
    ├── primitives              dict[str, PrimitiveHint]  §6
    ├── strategies              StrategyPreferences       §7
    ├── thresholds              dict[str, SeverityPriors] §8
    ├── mappings                list[MappingRule]         §9
    ├── constraints             GuardrailConstraints      §10
    ├── priorities              StrategyPriorities        §11
    ├── parameter_bias_rules    list[ParameterBiasRule]   §2.4
    ├── bias_semantics          dict[str, StrategyBiasSemantics]  §2.4.1
    ├── conflict_resolution     ConflictResolution        §2.4.3
    └── bias_application_rules  list[BiasApplicationRuleEntry]    §2.4.4

Design notes
────────────
MappingRule.condition has alias "if" to mirror the YAML key (which is a
  Python keyword). When parsing from YAML/dict, the key 'if' maps to the
  Python attribute `condition`. The GuardrailsLoader must ensure the raw
  dict uses 'if' as the key; Python code accesses it as `rule.condition`.

SeverityPriors uses float | None per strategy-severity pair because
  threshold strategy has null priors (explicit in the spec — no global
  prior means the LLM must request clarification).

GuardrailConstraints.threshold_bounds is keyed by strategy name string
  (not StrategyType enum) because guardrails are loaded independently of
  the condition model layer and must not create a circular import.

Guardrails.strategy_registry is non-empty at startup — ConfigApplyService
  raises ConfigError if strategy_registry is empty (§13 validation checklist
  in memintel.config.md).

ApplicationContext is imported from config.py because it is defined in
  py-instructions.md alongside the ConfigSchema models and is also used
  directly by ConfigApplyService when storing the extracted context.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.config import ApplicationContext


# ── §3 Strategy Registry ──────────────────────────────────────────────────────

class StrategyParameterConstraints(BaseModel):
    """Bounds declared on a strategy parameter in the registry."""
    min: float | None = None
    max: float | None = None
    min_items: int | None = None    # for list-typed parameters (e.g. composite.operands)


class StrategyParameter(BaseModel):
    """
    Schema for a single parameter of a registered strategy.

    type identifies the value kind: float, enum, duration, list<string>,
    list<decision<boolean>>. The compiler validates actual condition params
    against these definitions.

    values is only populated for enum-typed parameters (e.g. direction).
    default is only populated for optional parameters with a declared default.
    """
    type: str
    required: bool = True
    values: list[str] | None = None         # enum values, e.g. [above, below]
    constraints: StrategyParameterConstraints | None = None
    default: str | None = None
    description: str | None = None


class StrategyRegistryEntry(BaseModel):
    """
    A registered strategy definition from the strategy_registry block.

    input_types lists the Memintel types this strategy accepts as its
    primary concept input (e.g. float, time_series<float>).

    output_type is always one of: decision<boolean>, decision<categorical>.

    calibration_note and bias_note carry advisory text for the LLM;
    they are not enforced by the compiler.
    """
    version: str
    description: str
    input_types: list[str]
    output_type: str
    parameters: dict[str, StrategyParameter] = Field(default_factory=dict)
    example: str | None = None
    calibration_note: str | None = None
    bias_note: str | None = None


# ── §4 Type-Strategy Compatibility ────────────────────────────────────────────

class TypeCompatibilityEntry(BaseModel):
    """
    Valid and invalid strategies for a given primitive type.

    Enforced at compile time — a condition that pairs a strategy with an
    incompatible primitive type raises type_error. The compiler imports
    TYPE_STRATEGY_COMPATIBILITY from condition.py as the single Python-side
    source of truth; this model is the parsed representation from the
    guardrails file for validation and audit.
    """
    valid_strategies: list[str]
    invalid_strategies: list[str]
    note: str | None = None


# ── §5 Severity Vocabulary ────────────────────────────────────────────────────

class SeverityLevel(BaseModel):
    """
    A named severity tier with its associated natural language signals.

    id is the canonical tier name: low, medium, high.
    natural_language_signals are the phrases the LLM matches to this tier.
    Severity resolution must be deterministic — same phrase, same tier.
    """
    id: str
    natural_language_signals: list[str] = Field(default_factory=list)


class SeverityResolutionRule(BaseModel):
    """
    Edge-case resolution rule for severity determination.

    Either supplies a default when no signal is found, or overrides the
    resolved severity when urgency signals are detected.

    Only one of (default_severity, override_severity) is set per rule.
    """
    if_no_severity_signal: bool | None = None
    default_severity: str | None = None
    if_urgency_signal: bool | None = None
    signals: list[str] | None = None          # urgency signal phrases
    override_severity: str | None = None


class SeverityVocabulary(BaseModel):
    """
    Complete severity vocabulary: levels + resolution rules.

    levels defines all severity tiers and their natural language signals.
    resolution_rules handles edge cases (no signal found, urgency override).
    """
    levels: list[SeverityLevel]
    resolution_rules: list[SeverityResolutionRule] = Field(default_factory=list)


# ── §6 Primitives (Domain Context) ────────────────────────────────────────────

class PrimitiveStrategyHints(BaseModel):
    """
    Domain hints about which strategies to prefer or avoid for a primitive.

    preferred   — bias selection toward these strategies when multiple are valid
    discouraged — lower priority; avoid unless strongly justified by intent

    These are soft hints for the LLM, not hard constraints. The hard constraint
    is type_compatibility (§4). A preferred strategy that is type-incompatible
    raises type_error at compile time regardless of this hint.
    """
    preferred: list[str] = Field(default_factory=list)
    discouraged: list[str] = Field(default_factory=list)


class PrimitiveHint(BaseModel):
    """
    Domain-specific metadata for a primitive, used to guide the LLM.

    threshold_priors is keyed by strategy name, then severity tier:
      {"change": {"low": 0.02, "medium": 0.05, "high": 0.10}, ...}
    A None value at a severity tier means no prior is defined — the LLM
    must request clarification rather than invent a value.

    version is the hint version string (informational, not enforced).
    """
    version: str = "1.0"
    type: str
    description: str | None = None
    strategy_hints: PrimitiveStrategyHints | None = None
    threshold_priors: dict[str, dict[str, float | None]] | None = None


# ── §7 Global Strategy Preferences ───────────────────────────────────────────

class StrategyPreferences(BaseModel):
    """
    Global strategy selection preferences applied when no primitive hint exists.

    preferred   — bias selection toward these when no primitive hint is active
    discouraged — lower priority; avoid unless intent strongly implies them
    disabled    — must not be used; compiler rejects violations (treated same
                  as disallowed_strategies in constraints for compile-time enforcement)
    """
    preferred: list[str] = Field(default_factory=list)
    discouraged: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)


# ── §8 Threshold Priors (Global Defaults) ─────────────────────────────────────

class SeverityPriors(BaseModel):
    """
    Global threshold priors for a single strategy, keyed by severity tier.

    None at a tier means no global prior is defined for that severity level.
    When None, and no primitive-level prior exists, and the user has not
    supplied an explicit value, the LLM must request clarification.
    """
    low: float | None = None
    medium: float | None = None
    high: float | None = None


# ── §9 Mapping Rules ──────────────────────────────────────────────────────────

class MappingCondition(BaseModel):
    """
    The match condition for a mapping rule.

    signal_type: 'relative' | 'absolute' — matches on how the user phrases
      the measurement (e.g. "top users" vs "above 0.8")
    pattern: 'change' | 'anomaly' | 'classification' — matches on the
      conceptual pattern of the user's intent.
    """
    signal_type: str | None = None
    pattern: str | None = None


class MappingRule(BaseModel):
    """
    Intent → strategy bias rule.

    Acts as bias, not strict enforcement. Overridden by user_explicit values
    and primitive-level hints (see StrategyPriorities for the full order).

    The YAML key 'if' maps to the Python attribute 'condition' to avoid
    shadowing the Python keyword. GuardrailsLoader must pass the dict with
    key 'if'; this model uses alias="if" for correct YAML round-tripping.

    prefer is the strategy name to bias toward when this rule matches.
    """
    model_config = {"populate_by_name": True}

    condition: MappingCondition = Field(alias="if")
    prefer: str
    rationale: str | None = None


# ── §10 Constraints (Hard Overrides) ─────────────────────────────────────────

class ThresholdBounds(BaseModel):
    """
    Hard min/max bounds for a strategy's primary numeric parameter.

    Enforced by CalibrationService and the compiler. Priors can be within
    or outside bounds — bounds cannot be crossed regardless.

    min=None means no lower bound. max=None means no upper bound.
    """
    min: float | None = None
    max: float | None = None


class MaxComplexity(BaseModel):
    """Hard limits on condition composition within a single task."""
    max_conditions_per_task: int = 3


class StrategyVersionPolicy(BaseModel):
    """
    Version constraints on strategy registry entries used in conditions.

    minimum_version — reject conditions referencing strategy versions below this.
    allow_deprecated — when False, compiler rejects deprecated strategy versions.
    """
    minimum_version: str = "1.0"
    allow_deprecated: bool = False


class GuardrailConstraints(BaseModel):
    """
    Hard overrides enforced by the compiler and runtime, regardless of LLM
    output or application context instructions.

    disallowed_strategies — strategy types that must not be used in any
      condition definition. Compiler raises semantic_error on violation.

    disallowed_primitives — primitive names that must not appear in any
      concept definition. Compiler raises reference_error on violation.

    threshold_bounds — per-strategy [min, max] bounds for the primary numeric
      parameter. Calibration and parameter bias must respect these bounds;
      on_bounds_exceeded governs what happens when a value would breach them.

    on_bounds_exceeded:
      clamp  — silently clamp to the nearest bound (default)
      reject — raise bounds_exceeded error (HTTP 400)
    """
    disallowed_strategies: list[str] = Field(default_factory=list)
    disallowed_primitives: list[str] = Field(default_factory=list)
    max_complexity: MaxComplexity = Field(default_factory=MaxComplexity)
    strategy_version_policy: StrategyVersionPolicy = Field(
        default_factory=StrategyVersionPolicy
    )
    threshold_bounds: dict[str, ThresholdBounds] = Field(default_factory=dict)
    on_bounds_exceeded: str = "clamp"   # clamp | reject


# ── §11 Strategy Selection Priority ──────────────────────────────────────────

class StrategyPriorities(BaseModel):
    """
    Ordered list of strategy selection priority layers.

    Earlier entries in `order` win. The first layer that resolves a strategy
    is used; lower layers are not consulted.

    Default order (from spec §11):
      user_explicit       — always wins; bias rules never applied
      primitive_hint      — primitive-level strategy_hints
      mapping_rule        — intent-based mapping from signal type or pattern
      application_context — strategy bias from domain instructions
      global_preferred    — strategies.preferred list
      global_default      — threshold_priors + bias shift applied
    """
    order: list[str] = Field(default_factory=lambda: [
        "user_explicit",
        "primitive_hint",
        "mapping_rule",
        "application_context",
        "global_preferred",
        "global_default",
    ])


# ── §2.4 Parameter Bias Rules ─────────────────────────────────────────────────

class BiasEffect(BaseModel):
    """
    The deterministic effect of a matched parameter bias rule.

    direction:      relax_threshold | tighten_threshold
    severity_shift: integer shift to apply to the resolved severity tier.
      -1 = shift one tier lower (high→medium, medium→low)
      +1 = shift one tier higher (low→medium, medium→high)
      Shifts past the boundary tier (below low, above high) are clamped.
    """
    direction: str          # relax_threshold | tighten_threshold
    severity_shift: int     # -1, 0, +1


class ParameterBiasRule(BaseModel):
    """
    Maps a substring match in application context instructions to a
    deterministic severity shift and threshold direction.

    Applied by the LLM during POST /tasks only — never at execution time.
    Rules are applied to both primitive-level and global prior lookups,
    but NEVER when user_explicit values are present.
    """
    if_instruction_contains: str
    effect: BiasEffect


# ── §2.4.1 Bias Semantics ─────────────────────────────────────────────────────

class StrategyBiasSemantics(BaseModel):
    """
    Concrete meaning of 'relax' and 'tighten' for a specific strategy.

    Authoritative mapping per the spec:
      threshold:  relax=decrease_value,      tighten=increase_value
      percentile: relax=increase_percentile, tighten=decrease_percentile
      change:     relax=decrease_percentage, tighten=increase_percentage
      z_score:    relax=decrease_threshold,  tighten=increase_threshold
      equals:     relax=not_applicable,      tighten=not_applicable
      composite:  relax=not_applicable,      tighten=not_applicable

    The value 'not_applicable' means no shift is applied and no error is
    raised — bias rules are silently ignored for this strategy.
    """
    relax: str
    tighten: str


# ── §2.4.3 Conflict Resolution ────────────────────────────────────────────────

class ConflictResolutionRule(BaseModel):
    """A single conflict resolution outcome with its action and documentation."""
    action: str
    example: str | None = None
    description: str | None = None
    note: str | None = None


class ConflictResolution(BaseModel):
    """
    Rules for resolving conflicts when multiple bias rules match.

    additive_same_direction        — shifts in the same direction are summed
      (clamped at boundary tier)
    additive_opposing_different_magnitude — apply the net direction and magnitude
    opposing_same_magnitude        — equal-and-opposite shifts are neutralized;
      no shift is applied (distinct from 'no rules matched')
    """
    additive_same_direction: ConflictResolutionRule | None = None
    additive_opposing_different_magnitude: ConflictResolutionRule | None = None
    opposing_same_magnitude: ConflictResolutionRule | None = None


# ── §2.4.4 Bias Application Rules ─────────────────────────────────────────────

class BiasApplicationRuleDetail(BaseModel):
    """The action and documentation for a single bias application edge case."""
    description: str
    action: str
    note: str | None = None


class BiasApplicationRuleEntry(BaseModel):
    """
    One entry in the bias_application_rules list.

    Each entry handles one edge case that arises after shifts are applied.
    Only one condition key is set per entry.

    if_shifted_severity_has_no_prior — the shifted severity tier has no prior
      value for the selected strategy. Action: request_clarification.

    if_shifted_value_violates_bounds — the prior at the shifted tier is outside
      the threshold_bounds. Action: apply_constraints.on_bounds_exceeded.
    """
    if_shifted_severity_has_no_prior: BiasApplicationRuleDetail | None = None
    if_shifted_value_violates_bounds: BiasApplicationRuleDetail | None = None


# ── Top-level Guardrails container ────────────────────────────────────────────

class Guardrails(BaseModel):
    """
    The complete parsed and validated representation of `memintel.guardrails.md`.

    Loaded by GuardrailsStore at startup via ConfigApplyService. Stored in
    memory; no runtime re-parsing. GuardrailsStore.get() returns this object.

    Consumers and which fields they use:
      LLM (POST /tasks):   all fields except constraints (which is compiler-only)
      Compiler:            strategy_registry, type_compatibility, constraints
      CalibrationService:  constraints.threshold_bounds, constraints.on_bounds_exceeded

    Startup invariant: strategy_registry must be non-empty. ConfigApplyService
    raises ConfigError('strategy_registry is empty') if this check fails.
    """
    application_context: ApplicationContext
    strategy_registry: dict[str, StrategyRegistryEntry]
    type_compatibility: dict[str, TypeCompatibilityEntry] = Field(default_factory=dict)
    severity_vocabulary: SeverityVocabulary | None = None
    primitives: dict[str, PrimitiveHint] = Field(default_factory=dict)
    strategies: StrategyPreferences = Field(default_factory=StrategyPreferences)
    thresholds: dict[str, SeverityPriors] = Field(default_factory=dict)
    mappings: list[MappingRule] = Field(default_factory=list)
    constraints: GuardrailConstraints = Field(default_factory=GuardrailConstraints)
    priorities: StrategyPriorities = Field(default_factory=StrategyPriorities)

    # §2.4 — LLM-only; excluded from compiler validation
    parameter_bias_rules: list[ParameterBiasRule] = Field(default_factory=list)
    bias_semantics: dict[str, StrategyBiasSemantics] = Field(default_factory=dict)
    conflict_resolution: ConflictResolution | None = None
    bias_application_rules: list[BiasApplicationRuleEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_non_empty_strategy_registry(self) -> Guardrails:
        if not self.strategy_registry:
            raise ValueError(
                "strategy_registry is empty — at least one strategy must be "
                "registered. ConfigApplyService raises ConfigError on this "
                "condition."
            )
        return self
