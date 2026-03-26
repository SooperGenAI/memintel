---
id: patterns
title: Core Patterns
sidebar_label: Core Patterns
---

# Core Patterns

This section explains the common decision patterns used in agentic systems with Memintel. These patterns represent reusable ways to convert meaning into deterministic decisions — implemented through `Concept → Condition`, expressed via a strategy type and parameter definitions.

---

## Threshold Detection

**Purpose:** Trigger an action when a value crosses a fixed threshold.

```yaml
strategy:
  type: threshold
  params:
    value: 0.8
```

**Use cases:** Churn risk detection, fraud probability, system load thresholds.

Replaces non-deterministic checks like `if "risk seems high"` with a deterministic rule. Simple, interpretable, and consistent.

---

## Change Detection

**Purpose:** Detect significant changes over time.

```yaml
strategy:
  type: change
  params:
    percentage: 0.3
```

**Use cases:** Sudden drop in engagement, spike in errors, rapid change in metrics.

Useful when agents monitor evolving systems. Captures dynamics — detects shifts, not just absolute levels.

---

## Percentile-Based Detection

**Purpose:** Evaluate a value relative to a population.

```yaml
strategy:
  type: percentile
  params:
    value: 95
```

**Use cases:** Top-performing users, outlier detection, ranking-based triggers.

Useful when absolute thresholds are not meaningful. Provides relative evaluation that adapts to distribution changes.

---

## Z-Score Anomaly Detection

**Purpose:** Detect statistical anomalies relative to a baseline.

```yaml
strategy:
  type: z_score
  params:
    threshold: 2
```

**Use cases:** Anomaly detection, unusual system behavior, abnormal financial activity.

Statistically grounded and robust to noise. Allows agents to react to unexpected changes.

---

## Trend Detection

**Purpose:** Identify consistent upward or downward movement.

```yaml
strategy:
  type: change
  params:
    direction: "up"
    window: "7d"
```

**Use cases:** Growth trends, declining engagement, performance degradation.

Captures direction — not just magnitude. Helps agents respond to gradual changes.

---

## Divergence Detection

**Purpose:** Detect mismatch between two signals.

```yaml
strategy:
  type: composite
  params:
    expression: "narrative_signal - price_signal > threshold"
```

**Use cases:** Narrative vs behavior mismatch, expectation vs reality gaps, model vs actual divergence.

Critical for systems where LLM perception might differ from real-world data. Surfaces hidden risk.

---

## Composite Conditions

**Purpose:** Combine multiple conditions into a single decision.

```yaml
strategy:
  type: composite
  params:
    expression: "(high_risk AND high_value) OR critical_event"
```

**Use cases:** Multi-factor decisions, prioritization logic, complex workflows.

Allows structured decision logic without relying on LLM reasoning. Composable, expressive — and still fully deterministic.

---

## Pattern Selection Guide

| Signal type | Recommended pattern |
|---|---|
| Stable absolute metrics | `threshold` |
| Dynamic / evolving systems | `change` |
| Relative ranking needed | `percentile` |
| Statistical anomalies | `z_score` |
| Categorical matching | `equals` |
| Multiple factors combined | `composite` |

---

## Combining Patterns

Patterns can be layered for higher precision:

```yaml
strategy:
  type: composite
  params:
    expression: "z_score > 2 AND change > 0.2"
```

This allows higher precision, reduced false positives, and better overall decision quality.

---

## Key Principles

1. Always separate computation (concept) from evaluation (condition)
2. Express all decisions using strategies and parameters
3. Choose the simplest pattern that works
4. Prefer deterministic strategies over heuristic reasoning
5. Never embed decision logic inside agent prompts
