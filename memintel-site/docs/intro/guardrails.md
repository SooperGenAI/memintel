---
id: guardrails
title: Guardrails System
sidebar_label: Guardrails System
---

# Guardrails System

Memintel enforces correctness and consistency through a structured **Guardrails System**. Guardrails define the constraints, compatibility rules, and domain boundaries within which all concepts, conditions, and actions must operate.

They are defined in a separate file: **`memintel.guardrails.md`**

---

## Purpose

The guardrails system ensures that:
- All decision logic is structurally valid
- Strategies are used correctly
- Parameters are within valid bounds
- LLM-generated definitions cannot violate system constraints
- Decision behavior remains consistent across environments

Without guardrails, systems risk invalid condition definitions, incompatible strategy usage, unstable parameter selection, and non-reproducible decisions.

---

## Core Components

### 1. Strategy Registry

A centralised registry of all supported decision strategies. Every strategy — `threshold`, `percentile`, `z_score`, `change`, `equals`, `composite` — is a versioned object with a declared input type, parameter schema, and output type. The LLM can only select strategies that exist in this registry.

### 2. Type–Strategy Compatibility

A compatibility map enforces which strategies are valid for each primitive type.

| Strategy | Valid Input Types |
|---|---|
| `threshold` | `float`, `int` |
| `percentile` | `float`, `int` |
| `z_score` | `time_series<float>`, `time_series<int>` |
| `change` | `time_series<float>`, `time_series<int>` |
| `equals` | `string`, `categorical` |
| `composite` | Composed from other conditions |

Incompatible pairings are rejected at compile time.

### 3. Parameter Constraints

Defines valid ranges and structures for strategy parameters.

| Strategy | Parameter | Constraint |
|---|---|---|
| `threshold` | `value` | Within declared bounds |
| `percentile` | `value` | 0–100 |
| `z_score` | `threshold` | Must be > 0 |
| `change` | `value` | Within declared bounds |

These constraints prevent invalid configurations, ensure stable evaluation behavior, and enforce consistency across systems.

### 4. Domain Constraints

Defines application-specific rules and preferences — acceptable thresholds for certain signals, prioritization rules, and risk tolerance boundaries. These are derived from the application context and system requirements.

### 5. Hard Constraints vs Soft Guidance

| Type | Examples | Effect |
|---|---|---|
| **Hard Constraints** | Type compatibility, required parameters, valid ranges, structural correctness | Violations result in rejection |
| **Soft Guidance** | Preferred parameter ranges, recommended strategies, domain heuristics | Influences definition and calibration, but does not block execution |

---

## Strategy Selection Priority

When the LLM resolves strategy and parameters during task creation, it follows a strict priority order:

| Priority | Source | Description |
|---|---|---|
| 1 (highest) | `user_explicit` | Threshold or strategy explicitly provided by the user. Always wins. |
| 2 | `primitive_hint` | Strategy hints declared on the primitive in guardrails. |
| 3 | `mapping_rule` | Intent pattern matched to a strategy (e.g. `"rises"` → `change`). |
| 4 | `application_context` | Strategy bias from domain instructions. |
| 5 | `global_preferred` | Globally preferred strategies declared in guardrails. |
| 6 (fallback) | `global_default` | Global threshold priors. |

The same intent + same guardrails always produces the same strategy and parameters. This is not heuristic inference — it is deterministic compilation.

---

## Relationship to Application Context

| | Role |
|---|---|
| **Application context** | Provides domain understanding, instructions, intent biasing |
| **Guardrails** | Provides enforcement, validation, structural constraints |

Together: application context guides interpretation, guardrails ensure correctness.

---

## Role in the System

Guardrails operate at **definition and validation time** — when concepts are created, when conditions are defined, when strategies and parameters are assigned. They ensure that all executable logic is valid, consistent, and deterministic.

At runtime, guardrails are not re-evaluated dynamically — they are already enforced through validated definitions.

```
Application Context → guides intent
Guardrails          → constrain interpretation
Concept             → computes meaning
Condition           → evaluates via strategy
Action              → executes
```

---

## IR Hash and Determinism Guarantee

The `ir_hash` is a SHA-256 hash of the **normalised** execution graph. Normalisation ensures that semantically identical definitions always produce the same hash regardless of node insertion order, parameter representation (e.g. `Decimal('1.0')` vs `Decimal('1.00')`), list parameter ordering, or input slot type representation. Two definitions with the same `ir_hash` are guaranteed to produce identical execution results for identical inputs.

### Normalisation Rules

| Normalisation Rule | Detail |
|---|---|
| Node order | Nodes sorted by `node_id` before hashing |
| Edge order | Edges sorted by `(from, to, slot)` before hashing |
| Dict key order | All dict keys sorted alphabetically (`sort_keys=True`) |
| Decimal parameters | All `Decimal` values converted to `float` before serialisation |
| List parameters | All list-valued parameters sorted before serialisation |
| `input_slot` type | Always serialised as string, never integer |
| Metadata exclusions | `graph_id`, `ir_hash`, and `created_at` excluded from hash input |

A `409 Conflict` response is returned if the submitted definition produces an `ir_hash` that conflicts with an existing version — indicating a structural collision that must be resolved before the definition can be stored.

---

## Managing Environment Changes

When the external environment changes significantly — a regulatory update, a shift in what "high risk" means in a domain, a new signal type becoming available — the admin has two distinct levers.

### Lever 1 — Update the guardrails config

The admin modifies the guardrails YAML file — updating parameter priors, bias rules, strategy preferences, or type-strategy mappings. This is a config change, not a code change.

For example, if regulatory guidance tightens and "significant" in AML context should now map to a lower threshold:

```yaml
# Before
parameter_priors:
  transaction.value_vs_baseline_ratio:
    high_severity: { threshold: 15.0 }

# After — regulatory environment tightened
parameter_priors:
  transaction.value_vs_baseline_ratio:
    high_severity: { threshold: 10.0 }
```

### Lever 2 — Trigger recompilation of affected tasks

With updated guardrails loaded, the admin triggers recompilation of affected tasks — re-running the compiler against the original intent using the new policy constraints. The original intent string is preserved. The compiler derives new concepts and conditions from the same meaning, within the new guardrails.

```python
# Admin triggers recompilation of affected tasks
await adminClient.tasks.recompile(
    task_ids=["tsk_aml_watch", "tsk_credit_risk"],
    reason="Regulatory tightening — updating thresholds per new FATF guidance"
)
```

This produces new concept and condition versions. Tasks are not automatically rebound — the admin reviews the compiled output first and explicitly approves:

```
Admin reviews delta:
  Condition: transaction.high_aml_risk
  Threshold: 15.0x → 10.0x baseline
  Estimated impact: +8 alerts per day
  Tasks affected: 14

→ Approve and rebind   → Reject   → Review individual tasks
```

### The full workflow

```
External environment changes (regulation, market, policy)
        ↓
Admin updates guardrails YAML
        ↓
Admin deploys updated config (system reloads guardrails)
        ↓
Admin identifies affected tasks
        ↓
Admin triggers recompilation
        ↓
Compiler re-runs with:
  original intent (unchanged) + new guardrails (updated)
        ↓
New concept + condition versions produced
        ↓
Admin reviews delta and approves
        ↓
Tasks explicitly rebound to new versions
        ↓
New evaluation loop begins with updated logic
```

### Key properties of this workflow

**The original intent is preserved.** The task still says "alert me when a transaction shows unusual risk." The compiler re-derives what "unusual" means under the new guardrails. Users never touched a threshold.

**Old versions are immutable.** Recompilation creates new versions. Every decision made under the old version remains fully reproducible. Historical audit trails are unaffected.

**Rebinding is explicit.** Nothing changes silently. The admin reviews the impact before any task uses the new logic.

**Not all tasks are affected equally.** The admin selectively recompiles only tasks whose intent intersects with the changed guardrail dimension. A task monitoring latency degradation is unaffected by a change to AML transaction thresholds.

---

## Key Principles

1. All strategies must be defined in the strategy registry
2. All conditions must pass guardrail validation before execution
3. Type–strategy compatibility must always be enforced
4. Parameters must conform to defined schemas and bounds
5. Guardrails separate enforcement from interpretation
6. When the environment changes, update guardrails and recompile — never rewrite intent
