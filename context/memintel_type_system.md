# Memintel Type System Specification
**Version:** 1.1 (consolidated)  
**Status:** Authoritative — supersedes v1 spec and v1.1 addendum  
**Audience:** Compiler, SDK authors, platform teams

---

## 1. Purpose

The type system defines:

- The kinds of data Memintel can operate on
- How data flows through concepts
- Which operations are valid at each node
- How types are validated during compilation

It ensures that all computations are valid, deterministic, and predictable.

---

## 2. Design Principles

1. Minimal set of types
2. Strong validation at compile time
3. No implicit coercion — strict typing, with one explicit exception (int → float subtype)
4. Deterministic behavior
5. Extensible in future versions

---

## 3. Type Inventory

### 3.1 Scalar Types

| Type | Category | Description |
|------|----------|-------------|
| `float` | Scalar | Continuous numeric value. Used for scores, metrics, probabilities. |
| `int` | Scalar | Discrete numeric value. Subtype of `float` (see §4). |
| `boolean` | Scalar | `true` or `false`. |
| `string` | Scalar | Arbitrary UTF-8 text value. |
| `categorical` | Scalar (constrained) | A `string` constrained to a declared label set (see §3.5). |

### 3.2 Container Types

| Type | Description |
|------|-------------|
| `time_series<float>` | Ordered sequence of float values indexed by time. |
| `time_series<int>` | Ordered sequence of int values indexed by time. |
| `list<float>` | Unordered or ordered collection of float values. |
| `list<int>` | Collection of int values. |

> **Note:** `boolean`, `string`, and `categorical` are not valid type parameters for `time_series` or `list` in v1.1.

### 3.3 Nullable Modifier

Any type `T` can be wrapped as `T?` to indicate the value may be absent (`null`).

```
float?
boolean?
time_series<float?>
string?
categorical?
```

Non-nullable types (the default) cannot hold `null`. Assigning `null` to a non-nullable type is a `type_error`.

### 3.4 Decision Types

Produced exclusively as output from Condition evaluation. Not assignable to scalar types without explicit unwrapping.

| Type | Produced by |
|------|-------------|
| `decision<boolean>` | Threshold, percentile, z-score, change, composite strategies |
| `decision<categorical>` | Equals strategy on categorical/string input |

### 3.5 Parameter-Only Types

| Type | Classification | Description |
|------|----------------|-------------|
| `duration` | Compile-time parameter only | Time window literal (e.g. `7d`, `30d`). Not a DAG node output type. |

`duration` may only appear as an operator parameter argument. It cannot be the declared type of a primitive, feature, or concept output. Using `duration` as a node output type is a `type_error`.

**Valid duration suffixes:**

| Suffix | Unit | Example |
|--------|------|---------|
| `h` | Hours | `24h` |
| `d` | Days | `7d`, `30d` |
| `w` | Weeks | `4w` |
| `m` | Months | `3m` |
| `y` | Years | `1y` |

> **Constraint:** Duration values must be positive integers. Fractional durations (e.g. `1.5d`) are a `type_error`.

---

## 4. Subtype Rules

### 4.1 int → float (widening)

`int` is a strict subtype of `float`. An `int` value is always assignable where `float` is expected.

```
int       →  float        ✓  (widening, implicit)
float     →  int          ✗  type_error (narrowing requires explicit cast)
int       →  boolean      ✗  type_error
float     →  boolean      ✗  type_error
```

The same subtype relationship applies to container generics:

```
time_series<int>   →  time_series<float>   ✓
time_series<float> →  time_series<int>     ✗  type_error
list<int>          →  list<float>          ✓
list<float>        →  list<int>            ✗  type_error
```

### 4.2 T → T? (nullable widening)

A non-nullable type is always assignable to its nullable variant:

```
float    →  float?    ✓
boolean  →  boolean?  ✓
float?   →  float     ✗  type_error (requires explicit null handling)
```

### 4.3 Explicit Cast: float → int

To narrow `float` to `int`, use the explicit `to_int` operator. This operator truncates toward zero and raises a `type_error` at runtime if the input is non-finite (`NaN`, `Infinity`).

```
to_int(input: float) → int
```

---

## 5. categorical Type

### 5.1 Declaration

A concept or primitive producing a `categorical` output must declare its label set at definition time:

```yaml
output_type: categorical
labels: ["low", "medium", "high"]
```

### 5.2 Enforcement Rules

- The compiler validates that all execution paths produce only declared labels.
- Runtime values outside the declared label set are a `type_error`.
- Two `categorical` types with different label sets are **not** compatible, even if the underlying string values overlap.
- A `categorical` type is **not** assignable to `string` without an explicit `unwrap_categorical` operator.

### 5.3 Condition Compatibility

`categorical` inputs are valid only with the `equals` condition strategy:

```yaml
condition: segment_is_high_risk
  input: user_segment        # type: categorical
  strategy:
    type: equals
    value: "high"
```

Threshold, percentile, z-score, and change strategies applied to `categorical` or `string` inputs are a `type_error`.

### 5.4 string vs categorical

- `string` is unconstrained free text
- `categorical` is a constrained enum with declared label set

Only `categorical` guarantees closed-world validation.
`string` does not enforce value constraints.

---

## 6. Nullable Types and Missing Data

### 6.1 Default Nullability

Primitives with no `missing_data_policy` declared produce **nullable** output by default (`T?`). Operators that do not accept nullable inputs will cause a `type_error` unless a policy is set or an explicit null-handling operator is applied.

### 6.2 missing_data_policy → Type Resolution

The `missing_data_policy` on a primitive resolves its output nullability at compile time:

| Policy | Effect on output type | Notes |
|--------|-----------------------|-------|
| *(none)* | `T?` | Null propagates through the graph |
| `null` | `T?` | Explicit null passthrough; same as no policy |
| `zero` | `T` | Null replaced with `0`; output is non-nullable |
| `forward_fill` | `T` | Null replaced with last known value; non-nullable |
| `backward_fill` | `T` | Null replaced with next known value; non-nullable |

### 6.3 Null Propagation Rule

If any input to an operator is `T?`, the output is `T?` unless the operator explicitly handles null (i.e. is one of the null-handling operators below).

### 6.4 Null-Handling Operators

These operators consume nullable inputs and produce non-nullable outputs:

```
coalesce(input: T?, default: T) → T
drop_null(input: time_series<T?>) → time_series<T>
fill_null(input: time_series<T?>, value: T) → time_series<T>
```

---

## 7. Operator Type Rules

### 7.1 Input Validation

Each operator declares required input types. Passing a value of the wrong type is a `type_error`.

```
pct_change(input: time_series<float>) → float
```

```
# Invalid — scalar passed where time_series required
pct_change(0.5)  →  type_error
```

### 7.2 Output Type Propagation

Each operator produces a typed output that must satisfy the input constraints of the next node in the DAG.

```
time_series<float>  →  pct_change   →  float
float               →  z_score      →  float
float               →  [condition]  →  decision<boolean>
decision<boolean>   →  [action]     →  ActionResult
```

### 7.3 No Implicit Casting

With the sole exception of `int → float` widening, there is no implicit type coercion.

```
float ≠ time_series<float>      # add(time_series, float) → type_error
boolean ≠ float                  # no numeric coercion
```

### 7.4 Operator Signature Conventions

- Operators declared with `float` input accept `int` via subtype widening.
- Operators requiring strict integer semantics declare `int` explicitly and reject `float` inputs.

```
count(input: time_series<int>) → int       # int-specific; rejects float
mean(input: time_series<float>) → float    # accepts time_series<int> via widening
```

---

## 8. Primitive Type Assignment

Each primitive must declare a type. If the type is nullable (missing data possible), declare `T?` or set `missing_data_policy`.

```yaml
primitive:
  user.activity_count:
    type: time_series<float>
    missing_data_policy: zero       # output type resolves to time_series<float> (non-nullable)

  user.last_active_days:
    type: int
    missing_data_policy: forward_fill

  user.segment:
    type: categorical
    labels: ["new", "active", "at_risk", "churned"]
```

---

## 9. Condition Type Rules

### 9.1 Input Types by Strategy

| Strategy | Valid input types | Invalid input types |
|----------|-------------------|---------------------|
| `threshold` | `float`, `int` | `string`, `categorical`, `time_series`, `list`, `duration` |
| `percentile` | `float`, `int` | `string`, `categorical`, `time_series`, `list`, `duration` |
| `z_score` | `float`, `int` | `string`, `categorical`, `time_series`, `list`, `duration` |
| `change` | `float`, `int` | `string`, `categorical`, `time_series`, `list`, `duration` |
| `equals` | `categorical`, `string` | `float`, `int`, `boolean`, `time_series`, `list` |
| `composite` | `decision<boolean>` | All other types |

### 9.2 Output Types

| Strategy | Output type |
|----------|-------------|
| `threshold`, `percentile`, `z_score`, `change`, `composite` | `decision<boolean>` |
| `equals` | `decision<categorical>` |

### 9.3 decision<T> Is Not a Raw Scalar

`decision<boolean>` is not the same as `boolean`. A decision value carries provenance — condition id, version, and evaluation timestamp — alongside the raw value. It cannot be fed directly into a concept operator expecting `boolean` without explicit unwrapping:

```
unwrap_decision(input: decision<T>) → T    # extracts raw value; provenance is discarded
```

> **Design constraint:** Actions are bound to `decision<T>` outputs, not raw scalars. This is what makes action triggering declarative.

---

## 10. Compiler Behavior

The compiler must:

1. Assign a type to every node in the DAG.
2. Validate operator input/output type compatibility at each edge.
3. Propagate types through the full DAG in topological order.
4. Resolve nullability based on `missing_data_policy` declarations on primitives.
5. Validate `categorical` label sets and enforce label closure.
6. Enforce that `duration` does not appear as a node output type.
7. Enforce that `decision<T>` does not flow into operator inputs (only into action bindings or `unwrap_decision`).
8. Raise `type_error` on any violation. Raise `graph_error` on circular dependencies.

### Compiler Inference Example

```yaml
primitive:
  user.activity_count:
    type: time_series<float>
    missing_data_policy: forward_fill   # resolves to time_series<float> (non-nullable)

feature:
  activity_drop:
    op: pct_change
    input: user.activity_count
```

Compiler steps:
1. `user.activity_count` → `time_series<float>` (non-nullable, policy applied)
2. `pct_change` accepts `time_series<float>` → valid
3. `pct_change` output → `float`
4. `activity_drop` type = `float`

---

## 11. Error Types

| Error | When raised |
|-------|-------------|
| `type_error` | Invalid input type for an operator; incompatible assignment; null assigned to non-nullable; `duration` used as node output; `categorical` value outside declared label set; float→int without explicit cast |
| `reference_error` | Missing primitive, feature, or concept reference |
| `graph_error` | Circular dependency in concept or compute DAG |
| `semantic_error` | Definition fails semantic validation (missing description, inconsistent semantic_type, op incompatibility) |
| `parameter_error` | Invalid operator parameter value (e.g. fractional duration, out-of-range value) |

---

## 12. Type Rules — Consolidated Reference

| # | Rule |
|---|------|
| 1 | Every primitive, feature, and concept output has exactly one declared type. |
| 2 | Operators enforce input/output types at compile time. |
| 3 | No implicit casting except `int → float` (widening) and `T → T?` (nullable widening). |
| 4 | Types propagate through the DAG in topological order. |
| 5 | Conditions require scalar input (`float`, `int`, `categorical`, or `string` depending on strategy). |
| 6 | Conditions never accept `time_series`, `list`, or `duration` as input. |
| 7 | Conditions always produce `decision<T>`, never raw scalars. |
| 8 | `decision<T>` flows only to action bindings or `unwrap_decision`. |
| 9 | `duration` is a compile-time parameter only; it is not a DAG node output type. |
| 10 | Primitives with no `missing_data_policy` produce nullable (`T?`) output by default. |
| 11 | Null propagates through operators unless a null-handling operator resolves it. |
| 12 | `categorical` types must declare a closed label set; values outside it are a `type_error`. |
| 13 | Two `categorical` types with different label sets are not compatible. |

---

## 13. Future Extensions (Not in v1.1)

Do **not** implement in v1.1:

- `embeddings` type
- `probabilistic` types
- `custom` types
- `time_series<boolean>`
- `list<boolean>`
- `list<string>` / `list<categorical>`

---

## 14. Changelog

| Version | Changes |
|---------|---------|
| 1.1 | Added `string` and `categorical` types with label set enforcement. Formalised `int` as subtype of `float`. Widened `time_series` and `list` to support `<int>` parameter. Classified `duration` as parameter-only (not a DAG output type). Introduced nullable modifier `T?` and `missing_data_policy` type resolution. Defined `decision<T>` as condition output type. Added `unwrap_decision`, `coalesce`, `drop_null`, `fill_null` operators. |
| 1.0 | Initial release. Scalar types: `float`, `int`, `boolean`. Container types: `time_series<float>`, `list<float>`. Parameter type: `duration`. |

## 15. Type → Strategy Compatibility

Defines which strategies are valid for each input type.

```yaml
type_strategy_map:

  float:
    - threshold
    - percentile
    - z_score
    - change

  categorical:
    - equals

  string:
    - equals

  decision<boolean>:
    - composite

Rules
Strategy must accept input type
Compiler must reject invalid bindings

## 16. LLM Constraints (Hard Rules)

When generating feature graphs and conditions:

LLM MUST:

- Use only types defined in this specification
- Respect operator input/output type signatures
- Ensure type compatibility across all steps
- Ensure condition inputs conform to §9
- Ensure condition outputs are of type decision<T>
- Respect type_strategy_map when selecting strategies

LLM MUST NOT:

- Apply operators to incompatible types
- Assume implicit casting
- Generate condition inputs of type time_series, list, or duration
