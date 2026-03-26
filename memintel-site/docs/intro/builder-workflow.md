---
id: builder-workflow
title: Builder Workflow
sidebar_label: Builder Workflow
---

# Builder Workflow — End-to-End Agentic Integration

This section explains how to integrate Memintel into a real agentic AI system, step by step.

The goal is to move from:

```
LLM-driven workflows (non-deterministic)
```

to:

```
LLM (optional) → Memintel (deterministic decisions) → Agents (execution)
```

---

## The Complete Agentic Flow

In a typical agent system:
```
User Input → LLM → Tool Calls → Action
```

With Memintel:
```
User / Events → State → Concept → Condition → Action → Agent Execution
```

Memintel becomes the decision layer between reasoning and execution.

---

## Step 1 — Configure Environment

### Purpose
Define how Memintel connects to your LLM, data sources, and application context.

```yaml
# memintel.config.yaml
llm:
  provider: openai
  model: gpt-4
  api_key: YOUR_API_KEY

application_context:
  description: "User retention system"
  instructions:
    - "Prioritize early churn detection"
    - "Avoid false positives for highly active users"
```

### Application Context

Application context defines the domain, system goals, and operational instructions. It guides how definitions are generated and influences parameter interpretation — but does not introduce non-determinism into execution.

Instructions are translated into parameter adjustments using a defined severity vocabulary and explicit bias rules. For example:
- `"be conservative"` → higher thresholds / lower sensitivity
- `"detect early"` → lower thresholds / higher sensitivity

These mappings are explicitly defined, consistently applied, and independent of runtime LLM behavior.

---

## Step 2 — Define Primitives

### Purpose
Represent agent-observable state in a structured way.

```python
define_primitive({
    "id": "user_login_count_7d",
    "type": "number"
})

define_primitive({
    "id": "user_last_active_days",
    "type": "number"
})
```

These could come from your database, agent memory, event streams, or tool outputs.

:::tip Key Rule
If the agent can "see" it, it should be a primitive.
:::

---

## Step 3 — Define Concepts

### Purpose
Convert raw state into meaningful signals.

```python
define_concept({
    "id": "churn_risk",
    "inputs": [
        "user_login_count_7d",
        "user_last_active_days"
    ],
    "compute": "weighted_sum(login_count, last_active_days)"
})
```

Concepts can combine multiple signals, incorporate LLM-derived signals (optional), and normalize raw inputs.

:::tip Key Rule
Concepts define **meaning**, not decisions.
:::

---

## Step 4 — Define Conditions

### Purpose
Convert meaning into a reliable, deterministic decision.

```python
define_condition({
    "id": "high_churn_risk",
    "concept": "churn_risk",
    "strategy": {
        "type": "threshold",
        "params": {
            "value": 0.8
        }
    }
})
```

This replaces non-deterministic LLM-based evaluation:

```python
# BAD — LLM decides
if "user seems likely to churn":
    ...

# GOOD — deterministic strategy
# condition evaluates churn_risk.value against threshold 0.8
```

All conditions are validated through guardrails, which enforce compatibility between concept output and strategy type, valid parameter ranges, and structural correctness.

:::warning Key Rule
Conditions must **never** depend on LLM output at runtime. All decision logic must be expressed through a strategy type and parameter definition.
:::

---

## Step 5 — Define Actions

### Purpose
Connect Memintel decisions to your agent system.

```python
define_action({
    "id": "trigger_retention_agent",
    "type": "webhook",
    "endpoint": "/agents/retention"
})
```

In agentic systems, actions typically call other agents, invoke tools, trigger workflows, or send alerts. Actions are triggered by decisions — not raw values.

---

## Step 6 — Execute, Calibrate & Iterate

### Execution

```python
result = evaluateFull({
    "concept": "churn_risk",
    "condition": "high_churn_risk",
    "entity": "user_123"
})
```

What happens internally: `Concept → Condition → Action`

### Calibration Loop

Over time you will evaluate false positives and false negatives, adjust strategy parameters, and refine concepts for better signal quality. Calibration results in updated condition versions with improved decision accuracy.

### Versioning & Rebinding

All changes are versioned. When conditions evolve, new versions are created and existing workflows can explicitly rebind to updated versions. This ensures no silent behavior changes and full auditability.

---

## Full System Flow

```
LLM (optional reasoning)
        ↓
Primitives (state)
        ↓
Concepts (meaning)
        ↓
Conditions (deterministic decision via strategies)
        ↓
Actions (trigger agents/tools)
        ↓
Agent execution
```

---

## Critical Design Principles

| Principle | Rule |
|---|---|
| Separate reasoning from decision | LLM → flexible reasoning. Memintel → deterministic decision. Never mix them. |
| Treat agents as executors | Agents should execute actions and follow instructions — not decide or interpret. |
| Centralise decision logic | All decision logic must live in Concepts + Conditions, not in prompts or agent code. |
| Use structured strategies | All decisions must be expressed through strategy types and parameter definitions. |
| Design for replayability | Every decision should be reproducible, auditable, and explainable. |
| Constrain with guardrails | All definitions are valid, strategies are compatible, parameters are within bounds. |

:::note Key Insight
Memintel is not replacing agents — it is making them reliable. It transforms agent systems from **reasoning-driven** to **decision-driven**.
:::
