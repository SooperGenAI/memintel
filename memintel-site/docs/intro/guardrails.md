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

## Key Principles

1. All strategies must be defined in the strategy registry
2. All conditions must pass guardrail validation before execution
3. Type–strategy compatibility must always be enforced
4. Parameters must conform to defined schemas and bounds
5. Guardrails separate enforcement from interpretation
