---
id: quickstart
title: Quickstart
sidebar_label: Quickstart (5 min)
---

# Quickstart — 5-Minute Agentic Integration

Build your first deterministic decision loop for an agentic system.

**Goal:** Take an LLM-driven signal → evaluate it deterministically → trigger an action.

---

## What You're Building

A simple agentic workflow:

```
User behavior → LLM signal → Memintel decision → Action
```

**Example:**
- LLM estimates churn risk
- Memintel evaluates: is this high risk?
- System triggers a retention action

---

## Recommended Setup Order

Before creating your first task, follow this sequence to get the most accurate results from day one:

:::tip
Defining application context first gives the LLM the domain knowledge it needs to generate more accurate concept and condition definitions. Without context, task creation still works but produces generic definitions that may need more calibration cycles to reach production accuracy.
:::

| Step | Action | Endpoint |
|---|---|---|
| **1** (Recommended) | Define application context | `POST /context` |
| **2** | Register primitives | `POST /definitions` (primitives) |
| **3** | Create tasks | `POST /tasks` |
| **4** | Execute | `POST /execute/full` |

See the [Application Context](/docs/intro/application-context) page for full details on configuring context before you begin.

---

## Step 1 — Minimal Setup

Create a config file:

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

Even if your agent already uses an LLM, Memintel needs this for agent-assisted definition and semantic validation. The `application_context` guides how definitions are interpreted — without affecting deterministic execution.

---

## Step 2 — Define a Primitive

In agentic systems, primitives often come from logs, embeddings, LLM outputs, or user events.

```python
define_primitive({
    "id": "user_activity_score",
    "source": "app",
    "type": "number"
})
```

This represents a signal your agent already has.

---

## Step 3 — Define a Concept

Here's the key shift: even if an LLM produces a signal, Memintel reifies it as a deterministic concept.

```python
define_concept({
    "id": "churn_risk",
    "inputs": ["user_activity_score"],
    "compute": "normalize(activity_score)"
})
```

In more advanced cases, a concept can combine LLM output with system data, or be derived entirely from structured inputs.

---

## Step 4 — Define a Condition

This is where Memintel solves indeterminacy. **No LLM here.**

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

This is deterministic, reproducible, and auditable. Strategies ensure that decision logic is explicit and consistently applied.

---

## Step 5 — Define an Action

Actions connect Memintel to your agent system.

```python
define_action({
    "id": "trigger_retention_agent",
    "type": "webhook",
    "endpoint": "/agents/retention"
})
```

This could call another agent, trigger a workflow, or send a message. Actions are triggered based on evaluated decisions — not raw signals.

---

## Step 6 — Execute the Full Decision Loop

```python
result = evaluateFull({
    "concept": "churn_risk",
    "condition": "high_churn_risk",
    "entity": "user_123"
})
```

### Output

```python
result.value     # e.g. 0.87 — computed signal
result.decision  # True — deterministic evaluation
result.actions   # ["trigger_retention_agent"]
```

---

## What Just Happened

You created a hybrid system:

```
LLM       → Meaning     (flexible)
Memintel  → Decision    (deterministic)
Agent     → Execution
```

**Without Memintel:**
```python
# LLM decides directly — non-deterministic
if "user seems at risk":
    trigger_agent()
```
Problems: inconsistent decisions, no auditability, no reproducibility.

**With Memintel:**
```
LLM (optional) → Concept → Condition → Action
```
Meaning can evolve. Decisions remain stable. Interpretation is encoded through structured strategies and parameters — not embedded in prompts.

---

## Key Takeaway

:::note
LLMs generate signals. Memintel decides what those signals mean operationally.

Decision logic is **explicit**, **deterministic**, and **reusable**.
:::

In under 5 minutes, you've built a deterministic decision layer for your agent — a system that separates reasoning, decision-making, and execution.
