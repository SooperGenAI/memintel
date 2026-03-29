# Memintel Guardrails Specification
**File:** `memintel.guardrails.md`  
**Version:** 1.4.0  
**Scope:** App / Workspace Specific  
**Status:** Authoritative — supersedes v1.3.0

---

## Relationship to Other Specification Files

```text
memintel_type_system.md   → defines what is VALID (hard constraints, type rules)
memintel.guardrails.md    → defines application context and behavioral rules
memintel.config.md        → defines primitives and system configuration
```

> **Key Principle:** Guardrails guide behavior — they do NOT override correctness enforced by the type system or compiler.

---

## 0. Purpose

This document defines application-specific guardrails for Memintel. It serves as:

- The **application context layer** — domain description and LLM behavioral instructions
- The **parameter bias rules** — deterministic mappings from instructions to threshold shifts
- The **strategy registry** — defines all available strategies as versioned, typed objects
- The **type-strategy compatibility map** — declares which strategies are valid for which types
- The **LLM grounding layer** — soft constraints on strategy selection and parameter filling
- The **domain intelligence layer** — primitive-level hints and threshold priors
- The **severity vocabulary** — maps natural language intent to parameterized values
- The **behavior configuration** — priority ordering for strategy resolution

---

## 1. Structure Overview

```yaml
guardrails:
  application_context:    # domain description + LLM behavioral instructions
  strategy_registry:      # defines all strategies as versioned objects
  type_compatibility:     # maps types to valid strategies
  severity_vocabulary:    # maps natural language to severity levels
  primitives:             # domain-specific hints per primitive
  strategies:             # global strategy preferences
  thresholds:             # global threshold priors by strategy and severity
  mappings:               # intent → strategy bias rules
  constraints:            # hard app-level restrictions
  priorities:             # resolution order when multiple strategies are valid
```

---

## 2. Application Context

Provides semantic and domain-level grounding to the LLM during task generation. Defined exclusively in `memintel.guardrails.md`. Injected into the LLM system prompt at initialization alongside the guardrails.

> **Important:** Application context is non-binding. It influences but does not override guardrails or type system rules. It operates strictly at generation time (`POST /tasks`) and has zero influence on execution, evaluation, calibration, feedback, or condition logic post-compilation.

```yaml
application_context:

  name: "Acme SaaS Platform"

  description: >
    A B2B SaaS platform focused on user engagement and retention.
    Early detection of churn signals is critical. Users are typically
    customer success managers monitoring cohorts of business accounts.

  instructions:
    - Prefer early detection over precision when intent is ambiguous
    - Avoid missing negative trends — false negatives are more costly
      than false positives in this domain
    - Use conservative thresholds for drop and churn detection
    - Escalate high severity alerts via workflow actions, not notifications
    - When entity type is ambiguous, default to user accounts

  default_entity_scope: "user"

  action_preferences:
    high_severity: workflow
    medium_severity: notification
    low_severity: notification
    fallback:
      no_severity: notification     # used when severity cannot be determined
      categorical: notification     # used when condition output is decision<categorical>
```

### 2.1 Role of Application Context

| Component | Role | Binding? |
|-----------|------|----------|
| Type system | Hard structural constraints | Yes — compiler enforced |
| Guardrails constraints | Hard app-level restrictions | Yes — compiler enforced |
| Guardrails preferences | Strategy and threshold guidance | Soft — LLM bias |
| Application context | Semantic domain guidance | No — LLM influence only |

### 2.2 What Application Context Influences

- **Strategy bias** — steers strategy selection when multiple valid options exist; cannot override `disallowed_strategies` or `type_compatibility` rules
- **Parameter bias** — instructions are mapped to deterministic severity shifts via `parameter_bias_rules` (§2.4); not interpreted freely by the LLM
- **Action resolution** — `action_preferences` provides a domain-level default at position 3 in the action binding resolution order (between guardrails default and system default); subject to constraints (see §2.2.1)
- **Entity interpretation** — `default_entity_scope` anchors entity resolution when intent is ambiguous
- **Explanation terminology** — application context may influence domain vocabulary in natural language summaries; it must not influence decision values, parameter values, or attribution weights (see §2.3)

#### 2.2.1 Action Preferences and Constraints

Action preferences are subject to the constraints defined in §10. If a preferred action type is disallowed, the system falls back to the next layer in the action resolution order (system default). Application context cannot nominate a disallowed action.

```text
Action resolution order:
  1. user_explicit
  2. guardrails default
  3. application_context.action_preferences  ← subject to constraints
  4. system default
  5. fail if unresolved
```

### 2.3 Explanation Determinism Rules

The explain APIs (`POST /conditions/explain`, `POST /decisions/explain`) are guaranteed deterministic. Application context must not compromise this guarantee.

```yaml
explanation_rules:

  deterministic: true

  allowed_influences:
    - terminology         # domain-specific vocabulary in natural language summaries
    - domain_vocabulary   # e.g. "user account" instead of "entity"

  disallowed_influences:
    - parameter_values    # threshold, percentile cutoff, change %, z-score threshold
    - decision_outcome    # whether the decision is true/false or which label matched
    - attribution_weights # contribution of each input signal to the concept value
```

> **Rule:** Application context may influence wording but MUST NOT influence decision value, parameter values, or attribution weights. Explanation outputs must be fully reproducible given the same condition version, entity, and timestamp regardless of application context.

### 2.4 Parameter Bias Rules

Converts application context instructions into deterministic severity shifts. This replaces implicit LLM interpretation with an explicit, auditable transformation. The same instruction always produces the same shift.

`severity_shift: -1` means shift one tier lower (high → medium, medium → low).  
`severity_shift: +1` means shift one tier higher (low → medium, medium → high).  
Shifts that would go below `low` or above `high` are clamped at the boundary.

```yaml
parameter_bias_rules:

  - if_instruction_contains: "early detection"
    effect:
      direction: relax_threshold
      severity_shift: -1

  - if_instruction_contains: "avoid missing"
    effect:
      direction: relax_threshold
      severity_shift: -1

  - if_instruction_contains: "avoid noise"
    effect:
      direction: tighten_threshold
      severity_shift: +1

  - if_instruction_contains: "conservative thresholds"
    effect:
      direction: tighten_threshold
      severity_shift: +1

  - if_instruction_contains: "conservative"
    effect:
      direction: tighten_threshold
      severity_shift: +1
```

**Scope of application:**

Parameter bias rules apply whenever parameter values are derived from priors — both primitive-level and global. They do NOT apply when `user_explicit` values exist. `user_explicit` always wins and bias rules are never applied.

```text
Bias rules apply to:
  - primitive-level threshold_priors
  - global threshold_priors (thresholds block)

Bias rules do NOT apply to:
  - user_explicit values (user has provided an explicit threshold or strategy)
  - composite strategy parameters (see §2.4.2)
```

#### 2.4.1 Strategy-Specific Bias Semantics

`relax_threshold` and `tighten_threshold` have different concrete meanings per strategy. The following table is the authoritative mapping — it is what the system executes, not the LLM's interpretation.

```yaml
bias_semantics:

  threshold:
    relax:   decrease_value     # lower the cutoff → condition fires more easily
    tighten: increase_value     # raise the cutoff → condition fires less easily

  percentile:
    relax:   increase_percentile  # widen the qualifying population
    tighten: decrease_percentile  # narrow the qualifying population

  change:
    relax:   decrease_percentage  # smaller change required to trigger
    tighten: increase_percentage  # larger change required to trigger

  z_score:
    relax:   decrease_threshold   # fewer standard deviations required to trigger
    tighten: increase_threshold   # more standard deviations required to trigger

  equals:
    relax:   not_applicable       # no numeric parameter; see §2.4.2
    tighten: not_applicable

  composite:
    relax:   not_applicable       # see §2.4.2
    tighten: not_applicable
```

#### 2.4.2 Composite Strategy and Bias

Bias rules do NOT apply to `composite` strategies directly. Composite conditions have no numeric parameters of their own — they combine existing `decision<boolean>` outputs with a logical operator. Bias applies only to the underlying operand conditions when those are generated, not to the composite wrapper.

Similarly, `equals` strategy has no numeric parameter. Bias rules produce `not_applicable` for `equals` — no shift is applied, and no error is raised.

#### 2.4.3 Conflict Resolution

When multiple bias rules match the active instructions, shifts are summed. Conflicting directions are resolved as follows:

```yaml
conflict_resolution:

  additive_same_direction:
    action: sum_shifts
    example: "early detection" (-1) + "avoid missing" (-1) = -2, clamped to low

  additive_opposing_different_magnitude:
    action: apply_net_direction
    example: "early detection" (-1) + "conservative" (+1) = 0
    note: net zero means no shift; resolved severity is used as-is

  opposing_same_magnitude:
    action: neutralize
    description: >
      When two rules produce equal and opposite shifts, the result is
      explicitly neutralized — no shift is applied. This is distinct from
      a case where no bias rules matched at all (which also produces no
      shift but carries no conflict signal).
    example: "avoid missing" (-1) + "avoid noise" (+1) = neutralized
```

#### 2.4.4 Shifted Severity Resolution

After all shifts are computed and applied, the resulting severity tier is used to look up a threshold prior. Two edge cases must be handled:

```yaml
bias_application_rules:

  - if_shifted_severity_has_no_prior:
      description: >
        The shifted severity tier exists in severity_vocabulary but has no
        corresponding prior in threshold_priors for the selected strategy
        (e.g. threshold strategy where all priors are null).
      action: request_clarification
      note: Do not invent a value. Do not silently use a neighboring tier.

  - if_shifted_value_violates_bounds:
      description: >
        The prior value at the shifted severity tier falls outside the
        threshold_bounds declared in constraints.
      action: apply_constraints.on_bounds_exceeded
      note: >
        Apply the on_bounds_exceeded policy (clamp or reject) defined in
        §10. The bias shift does not override bounds.
```

### 2.5 LLM Influence Boundary

Application context operates strictly at generation time (`POST /tasks`).

```text
It has ZERO influence on:
  - execution          (POST /evaluate/full)
  - condition logic    (post-compilation)
  - evaluation results
  - calibration        (POST /conditions/calibrate)
  - feedback           (POST /feedback/decision)
  - explanation values (decision outcome, parameters, attribution)
```

### 2.6 Injection Order into LLM Context

```text
[1] Type system summary       → hard rules (what is valid)
[2] Guardrails                → strategy registry, compatibility, priors
[3] Application context       → domain description and instructions
[4] Parameter bias rules      → deterministic instruction → shift mappings
[5] Primitive registry        → available data signals
```

---

## 3. Strategy Registry

Defines all available strategies as reusable, versioned objects. Each strategy declares its input type requirements, parameter schema, and evaluation semantics.

```yaml
strategy_registry:

  threshold:
    version: "1.0"
    description: >
      Evaluates whether a scalar value is above or below a fixed cutoff.
      Use for absolute measurements where a meaningful boundary exists.
    input_types:
      - float
      - int
    output_type: decision<boolean>
    parameters:
      direction:
        type: enum
        values: [above, below]
        required: true
      value:
        type: float
        required: true
        constraints:
          min: null
          max: null
    example: "Is churn_score above 0.8?"

  percentile:
    version: "1.0"
    description: >
      Evaluates whether a value falls within the top or bottom N% of a
      reference population. Use when absolute values are not meaningful
      and relative ranking matters.
    input_types:
      - float
      - int
    output_type: decision<boolean>
    parameters:
      direction:
        type: enum
        values: [top, bottom]
        required: true
      value:
        type: float
        required: true
        constraints:
          min: 0.0
          max: 100.0
    example: "Is activity_count in the bottom 10% of users?"

  change:
    version: "1.0"
    description: >
      Evaluates whether a value has changed by more than X% relative to
      a previous value or window average. Use for detecting movement in
      time-series data.
    input_types:
      - time_series<float>
      - time_series<int>
    output_type: decision<boolean>
    parameters:
      direction:
        type: enum
        values: [increase, decrease, any]
        required: true
      value:
        type: float
        required: true
        constraints:
          min: 0.0
          max: null
      window:
        type: duration
        required: false
        default: "1d"
    example: "Has stock.price increased by more than 10% in the last 1d?"

  z_score:
    version: "1.0"
    description: >
      Evaluates whether a value deviates from its historical mean by more
      than N standard deviations. Use for anomaly detection on stable,
      normally-distributed signals.
    input_types:
      - time_series<float>
      - time_series<int>
    output_type: decision<boolean>
    parameters:
      threshold:
        type: float
        required: true
        constraints:
          min: 0.0
          max: null
      direction:
        type: enum
        values: [above, below, any]
        required: true
      window:
        type: duration
        required: false
        default: "30d"
    example: "Is revenue_growth more than 2 standard deviations above the 30d mean?"

  equals:
    version: "1.0"
    description: >
      Evaluates whether a categorical value matches one or more declared
      labels. Use for classification outputs and status-based conditions.
    input_types:
      - categorical
      - string
    output_type: decision<categorical>
    parameters:
      value:
        type: string
        required: true
      labels:
        type: list<string>
        required: false
        description: >
          If provided, the value must be a member of this list.
          Must be a subset of the categorical type's declared label set.
    calibration_note: >
      equals has no numeric parameter to adjust. Feedback submitted against
      an equals condition always returns no_recommendation from calibration.
      Logic changes require creating a new condition version.
    bias_note: >
      Parameter bias rules produce not_applicable for equals.
      No shift is applied and no error is raised.
    example: "Is user_segment equal to 'high_risk'?"

  composite:
    version: "1.0"
    description: >
      Combines two or more decision<boolean> outputs using logical
      operators (AND, OR). Use for multi-factor conditions.
    input_types:
      - decision<boolean>
    output_type: decision<boolean>
    parameters:
      operator:
        type: enum
        values: [AND, OR]
        required: true
      operands:
        type: list<decision<boolean>>
        required: true
        constraints:
          min_items: 2
    bias_note: >
      Parameter bias rules do not apply to composite strategies directly.
      Bias applies only to the underlying operand conditions when generated.
    example: "Is churn_risk high AND payment_risk high?"
```

---

## 4. Type-Strategy Compatibility Map

Defines which strategies are valid for each primitive type. Enforced at compile time.

> **Compiler rule:** If a guardrail file declares a primitive-level `strategy_hints` entry referencing a strategy incompatible with the primitive's declared type, the compiler raises a `type_error` and rejects the file on load.

```yaml
type_compatibility:

  float:
    valid_strategies:   [threshold, percentile]
    invalid_strategies: [change, z_score, equals, composite]

  int:
    valid_strategies:   [threshold, percentile]
    invalid_strategies: [change, z_score, equals, composite]

  time_series<float>:
    valid_strategies:   [change, z_score, percentile, threshold]
    invalid_strategies: [equals, composite]

  time_series<int>:
    valid_strategies:   [change, z_score, percentile, threshold]
    invalid_strategies: [equals, composite]

  categorical:
    valid_strategies:   [equals]
    invalid_strategies: [threshold, percentile, change, z_score, composite]

  string:
    valid_strategies:   [equals]
    invalid_strategies: [threshold, percentile, change, z_score, composite]

  boolean:
    valid_strategies:   []
    invalid_strategies: [threshold, percentile, change, z_score, equals, composite]
    note: >
      boolean primitives are not valid condition inputs directly.
      Use a concept to derive a float or categorical signal first.
```

---

## 5. Severity Vocabulary

Defines the severity levels recognised by the system and maps natural language signals to them. Severity is resolved first; parameter bias rules (§2.4) are applied as a separate subsequent step.

```yaml
severity_vocabulary:

  levels:
    - id: low
      natural_language_signals:
        - "slightly"
        - "a little"
        - "minor"
        - "small"
        - "modest"
        - "marginal"
    - id: medium
      natural_language_signals:
        - "noticeably"
        - "meaningfully"
        - "moderately"
        - "reasonably"
        - "somewhat"
    - id: high
      natural_language_signals:
        - "significantly"
        - "sharply"
        - "substantially"
        - "considerably"
        - "dramatically"
        - "major"
        - "large"

  resolution_rules:
    - if_no_severity_signal: true
      default_severity: medium
    - if_urgency_signal: true
      signals: ["immediately", "urgent", "critical", "emergency"]
      override_severity: high
```

> **LLM rule:** Severity resolution must be deterministic. Given the same input phrase and the same severity vocabulary, the LLM must always resolve to the same severity level. Parameter bias rules are applied after this resolution — they are a separate, subsequent step.

---

## 6. Primitives (Domain Context)

Defines primitives available to the LLM with domain-specific strategy hints and threshold priors. Each entry must reference only strategies compatible with the primitive's declared type (enforced against §4).

```yaml
primitives:

  stock.price:
    version: "1.0"
    type: time_series<float>
    description: Daily closing price of a stock
    strategy_hints:
      preferred: [change, percentile]
      discouraged: [z_score]
    threshold_priors:
      change:
        low: 0.02
        medium: 0.05
        high: 0.10
      percentile:
        low: 25
        medium: 10
        high: 5

  user.activity_count:
    version: "1.0"
    type: time_series<float>
    description: Daily user activity count
    strategy_hints:
      preferred: [percentile]
      discouraged: [z_score]
    threshold_priors:
      percentile:
        low: 25
        medium: 10
        high: 5
```

---

## 7. Global Strategy Preferences

```yaml
strategies:

  preferred:
    - percentile
    - change

  discouraged:
    - z_score

  disabled:
    - trend_change
```

- `preferred` → bias selection when no primitive hint exists
- `discouraged` → lower priority; avoid unless strongly justified
- `disabled` → must not be used; compiler rejects violations

---

## 8. Threshold Priors (Global Defaults)

Used when user intent is implicit and no primitive-level prior is defined.

```yaml
thresholds:

  change:
    low: 0.02
    medium: 0.05
    high: 0.10

  percentile:
    low: 25
    medium: 10
    high: 5

  threshold:
    low: null
    medium: null
    high: null

  z_score:
    low: 1.0
    medium: 2.0
    high: 3.0
```

> **Rule:** If a strategy has no global prior and no primitive-level prior, and the user has not supplied an explicit value, the LLM must request clarification. Do not invent a value.

> **Parameter bias interaction:** Bias rules shift which severity tier's prior is selected. They do not modify the prior values themselves. If the shifted tier has no prior, apply `bias_application_rules` (§2.4.4).

---

## 9. Mapping Rules (Intent → Strategy Bias)

Acts as bias, not strict enforcement. Overridden by explicit user input and primitive hints.

```yaml
mappings:

  - if:
      signal_type: relative
    prefer: percentile
    rationale: "Relative signals ('top users', 'bottom performers') imply ranking"

  - if:
      signal_type: absolute
    prefer: threshold
    rationale: "Absolute signals ('above 0.8', 'more than 100') imply fixed cutoff"

  - if:
      pattern: change
    prefer: change
    rationale: "Movement signals ('rises', 'drops', 'increases') imply change detection"

  - if:
      pattern: anomaly
    prefer: z_score
    rationale: "Anomaly signals ('unusual', 'abnormal', 'unexpected') imply z-score"

  - if:
      pattern: classification
    prefer: equals
    rationale: "Category signals ('is high_risk', 'labelled as') imply equals strategy"
```

---

## 10. Constraints (Hard Overrides)

Enforced by the compiler and runtime regardless of LLM output or application context instructions.

```yaml
constraints:

  disallowed_strategies:
    - z_score

  disallowed_primitives:
    - internal.test_metric

  max_complexity:
    max_conditions_per_task: 3

  strategy_version_policy:
    minimum_version: "1.0"
    allow_deprecated: false

  threshold_bounds:
    change:
      min: 0.01
      max: 0.25
    percentile:
      min: 1
      max: 50
    z_score:
      min: 0.5
      max: 4.0

  on_bounds_exceeded: clamp    # clamp | reject
```

**Threshold bounds rules:**
- Calibration and parameter bias must respect `threshold_bounds`
- Priors can be crossed; bounds cannot
- `clamp` — value is clamped to the nearest bound
- `reject` — operation returns a `bounds_exceeded` error

---

## 11. Strategy Selection Priority

```yaml
priorities:

  order:
    - user_explicit           # always wins; bias rules never applied
    - primitive_hint          # primitive-level strategy_hints
    - mapping_rule            # intent-based mapping (§9)
    - application_context     # strategy bias from instructions (§2)
    - global_preferred        # strategies.preferred list (§7)
    - global_default          # threshold_priors + bias shift applied (§8, §2.4)
```

| Priority | Description |
|----------|-------------|
| `user_explicit` | User-provided thresholds or strategy; bias rules never applied |
| `primitive_hint` | Domain-specific hint; bias rules applied to prior lookup |
| `mapping_rule` | Intent-based mapping from signal type or pattern |
| `application_context` | Strategy bias from domain instructions |
| `global_preferred` | Global preferred strategies list |
| `global_default` | Threshold priors fallback; bias rules applied to prior lookup |

> **Determinism rule:** Same input + same guardrails + same application context must always produce the same strategy selection and parameter values. `parameter_bias_rules` and `bias_semantics` guarantee this by converting instruction text and strategy type into fully deterministic transformations.

---

## 12. LLM Guidance Rules

**LLM SHOULD:**

```text
- Read application_context.description to understand the domain before
  processing any user intent
- Resolve application_context.instructions via parameter_bias_rules —
  do not interpret instruction text freely for parameter values
- Apply bias_semantics to translate relax/tighten direction into the
  correct concrete parameter change for the selected strategy
- Apply bias rules to both primitive-level and global prior lookups,
  unless user_explicit values are present
- Use conflict_resolution rules when multiple bias rules match
- Apply bias_application_rules when shifted severity has no prior or
  violates bounds
- Use application_context.default_entity_scope when entity type is ambiguous
- Use application_context.action_preferences for action resolution;
  use fallback.no_severity or fallback.categorical where applicable;
  skip to next resolution layer if preferred action is disallowed
- Consult strategy_registry before proposing a strategy
- Check type_compatibility before selecting a strategy for a given primitive
- Use severity_vocabulary to resolve severity, then apply parameter_bias_rules
- Prefer primitive-specific strategy hints when available
- Request clarification if no prior exists and user has not specified
```

**LLM MUST NOT:**

```text
- Freely interpret instruction text for parameter values outside of
  parameter_bias_rules
- Apply bias rules when user_explicit values are present
- Apply bias rules to composite strategy parameters directly
- Allow application_context to influence decision values, parameter values,
  or attribution weights in explanation outputs
- Allow application_context instructions to override disallowed_strategies
  or type_compatibility rules
- Allow application_context instructions to override user_explicit values
- Use disabled or disallowed strategies
- Select a strategy incompatible with the primitive's declared type
- Invent parameter values when no prior exists and user has not specified
- Reference strategies not defined in strategy_registry
- Apply application context at any stage other than POST /tasks generation
```

---

## 13. Example End-to-End

**Application context (active)**

```yaml
application_context:
  description: "B2B SaaS churn monitoring. Early detection is critical."
  instructions:
    - Prefer early detection over precision    # → severity_shift: -1
    - Use conservative thresholds              # → severity_shift: +1
```

**Bias computation:**
- "early detection" matches → `relax_threshold`, `severity_shift: -1`
- "conservative thresholds" matches → `tighten_threshold`, `severity_shift: +1`
- Opposing directions, equal magnitude → `conflict_resolution: neutralize`
- Net result: no shift applied

**Input**

```text
"Alert me when AAPL price rises significantly"
```

**Resolution steps**

```text
1.  Primitive identified    → stock.price  (type: time_series<float>)
2.  Type compatibility      → valid: change, z_score, percentile, threshold
3.  Intent pattern          → "rises" → pattern: change  (mapping rule)
4.  Primitive hint          → stock.price prefers change  (confirms)
5.  Severity signal         → "significantly" → high  (severity_vocabulary)
6.  Bias computation        → net shift = 0 (neutralized); severity = high
7.  Priority resolution     → primitive_hint wins
8.  Prior lookup            → change.high from stock.price priors → 0.10
9.  Bias semantics          → change + relax = decrease_percentage (not applied;
                              net shift is 0 and user has not been explicitly biased)
10. Bounds check            → 0.10 within change bounds [0.01, 0.25] ✓
11. Compiler validation     → change valid for time_series<float> ✓
                              value: 0.10 within parameter constraints ✓
                              strategy not disabled ✓
```

**Output condition**

```yaml
strategy: change
version: "1.0"
params:
  direction: increase
  value: 0.10
  window: 1d
severity: high
```

---

## 14. Interaction with Compiler

```text
Application context  →  LLM context only; excluded from compiler validation
Parameter bias rules →  LLM execution only; excluded from compiler validation
Guardrails           →  suggest, influence, bias, validate compatibility
Compiler             →  validate types, enforce constraints, guarantee correctness
```

The compiler consumes the guardrails file to:

1. Verify `strategy_hints` are compatible with primitive types (§4)
2. Reject conditions using `disabled` or `disallowed_strategies`
3. Validate strategy parameters against the schema in `strategy_registry`
4. Enforce `max_complexity` constraints
5. Enforce `strategy_version_policy`
6. Enforce `threshold_bounds` during calibration

The compiler does **not** consume `application_context`, `parameter_bias_rules`, `bias_semantics`, or `conflict_resolution`.

---

## 15. Versioning

| Increment | When |
|-----------|------|
| `MAJOR` | Strategy removed, compatibility rule changed, severity vocabulary restructured, bias_semantics structure changed, application context schema changed |
| `MINOR` | New strategy, new primitive, new mapping rule, new parameter_bias_rule, new bias_semantics entry, application context instructions updated |
| `PATCH` | Threshold tuning, description updates, natural language signal additions |

---

## 16. Key Design Principles

**Separation of concerns**

```text
Type system          → validity    (what is structurally correct)
Guardrails           → preference  (what is domain-appropriate)
Application context  → intent      (what this system is for)
Parameter bias rules → determinism (how intent maps to parameter shifts)
Bias semantics       → precision   (what relax/tighten means per strategy)
Compiler             → enforcement (what is actually executed)
```

**Deterministic override order**

Same input + same guardrails + same application context → same output. Parameter bias rules and bias semantics enforce this by converting free-text instruction semantics and strategy type into explicit, auditable transformations with defined conflict resolution.

**Hard boundary on LLM influence**

Application context and parameter bias rules operate only at `POST /tasks` generation time. Zero influence on execution, calibration, feedback, or explanation values.

**Config-driven behavior**

All strategy definitions, compatibility rules, severity mappings, threshold priors, parameter bias rules, bias semantics, and application instructions are declared explicitly. Nothing is hardcoded in the LLM prompt or compiler outside of what is defined here.

---

## 17. Summary

| Workflow Step | Covered by |
|---------------|------------|
| 0. Admin provides domain context | `application_context` (§2) |
| 1. Strategy definitions | `strategy_registry` (§3) |
| 2. Type-strategy compatibility | `type_compatibility` (§4) |
| 3. Admin sets guardrails | `primitives`, `strategies`, `thresholds` (§6–8) |
| 4. User expresses intent | Handled by LLM using §12 guidance rules |
| 5. LLM interprets intent | `severity_vocabulary` + `mappings` + `priorities` + `application_context` (§5, §9, §11, §2) |
| 6. Parameter filling | `threshold_priors` + `parameter_bias_rules` + `bias_semantics` + `conflict_resolution` + `bias_application_rules` (§8, §2.4) |
| 7. Action resolution | `application_context.action_preferences` + fallback + constraint check (§2.2.1) |
| 8. Compiler validation | `type_compatibility` + `constraints` + registry schema (§4, §10, §3) |
| 9. Condition created | Full output: strategy + version + params + severity |

> **One-line summary:** The guardrails file is the complete, deterministic contract between domain knowledge, LLM behavior, and compiler enforcement — with every instruction-to-parameter transformation explicitly defined, every conflict resolved by rule, and every influence boundary formally bounded.
