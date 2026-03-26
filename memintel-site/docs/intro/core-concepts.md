---
id: core-concepts
title: Core Concepts
sidebar_label: Core Concepts
---

# Core Concepts

To use Memintel effectively, you need to understand its core building blocks. Memintel introduces a structured, deterministic model of decision-making:

**Concept (ψ) → Condition (φ) → Action (α)**

Each layer has a clear responsibility — and must remain separate.

This model is supported by:
- **Primitives** as the interface to state
- **Features** as intermediate computations
- **Strategies** as structured evaluation logic
- **Parameters** as deterministic decision criteria
- **Guardrails** as constraints ensuring validity and consistency

---

## Primitives — Interface to Reality

Primitives are the **only** way Memintel accesses real-world data. They represent database fields, API responses, event streams, and metrics.

```yaml
primitive: user.activity_count
type: timeseries
entity: user_id
```

**Key properties:**
- Grounded in real data
- Strongly typed
- Deterministic and versioned
- Defined by the Builder

**Mental model:** `Reality → Primitives`

Primitives are facts, not interpretations.

---

## Features — Intermediate Signals

Features are intermediate computed values used within a concept to derive the final output. They transform primitives into more meaningful, structured signals.

```yaml
features:
  activity_drop:
    op: rate_of_change
    window: 7d
    input: user.activity_count
```

Here `user.activity_count` is a primitive, and `activity_drop` is a feature derived from it. The concept will use this feature to compute its final output.

**Key properties:**
- Deterministic
- Defined within a concept
- Composable (can depend on other features)
- Not independently executable

**Mental model:** `Primitives → Features → Concept Output`

:::note Important Distinctions
- **Primitive** → raw data
- **Feature** → derived signal
- **Concept** → final meaning

All intermediate computation inside a concept should be expressed as features.
:::

---

## Concepts (ψ) — Meaning (What is True)

Concepts define what should be computed from state. They transform primitives into meaningful signals.

```yaml
concept: churn_risk_v1
inputs:
  - user.activity_count
  - user.last_active_days
features:
  activity_drop:
    op: rate_of_change
    window: 7d
    input: user.activity_count
compute:
  churn_score:
    op: weighted_sum
    inputs:
      - activity_drop
      - user.last_active_days
```

**Key properties:**
- Deterministic
- Explicit (DSL-defined)
- Composable (can depend on other concepts)
- Versioned and immutable

**Mental model:** `Primitives → Concepts → Meaning (Rₜ)`

Concepts answer: **"What is happening?"**

---

## Conditions (φ) — Significance (What Matters)

Conditions define when a computed result is important enough to act on. They operate only on concept outputs and execute structured **strategies** — never ad-hoc rules.

```yaml
condition: high_churn_risk_v1
input: churn_risk_v1.churn_score
strategy:
  type: threshold
  params:
    value: 0.8
```

A more advanced example using statistical anomaly detection:

```yaml
condition: abnormal_risk_v1
input: churn_risk_v1.churn_score
strategy:
  type: z_score
  params:
    threshold: 2
```

**Available strategy types:**
- `threshold`
- `percentile`
- `z_score`
- `change`
- `equals`
- `composite`

Each strategy has a defined parameter schema, is versioned and validated, and is executed deterministically.

**Key properties:**
- Do NOT compute meaning — only interpret it
- Use structured strategies instead of implicit rules
- Deterministic and versioned
- Validated through guardrails

**Mental model:** `Meaning (Rₜ) → Condition → Decision (Aₜ)`

Conditions answer: **"Does this matter?"**

---

## Actions (α) — Execution (What to Do)

Actions define what happens when a condition is satisfied.

```yaml
action: send_retention_email
trigger:
  condition: high_churn_risk_v1
execution:
  type: webhook
  endpoint: /notify
```

**Key properties:**
- Triggered by decisions, not raw values
- No embedded logic
- Deterministic mapping from decision → execution
- Can integrate with APIs, workflows, and agents

**Mental model:** `Decision (Aₜ) → Action → System Behavior`

Actions answer: **"What should we do?"**

---

## Putting It All Together

Memintel enforces a strict pipeline:

```
Primitives → Concepts → Conditions → Actions
```

Formally:

```
Rₜ = ψ(Sₜ, Eₜ, C)   — concept computes meaning from state
Aₜ = φ(Rₜ, K)        — condition evaluates significance
Actionₜ = α(Aₜ)      — action maps decision to execution
```

---

## Why This Separation Matters

Most systems mix everything together:

```python
# BAD: mixed concerns
if churn_score > 0.8:
    send_email()
```

Memintel enforces: **Compute → Interpret → Act**

This gives you:
- **Determinism** — same input → same output
- **Reusability** — concepts reused across systems
- **Explainability** — every decision is traceable
- **Consistency** — all agents use the same logic

---

## Key Principles

| Principle | Rule |
|---|---|
| No hidden logic | All logic must be explicit, defined in DSL, and versioned |
| Meaning ≠ Decision | Concept computes value; Condition interprets value. Never mix them |
| Decisions use strategies | All decisions must be expressed as Strategy + Parameters |
| Agents do not decide | Agents can suggest definitions; Memintel executes decisions deterministically |
| Everything is versioned | Concepts, Conditions, Strategies, Parameters — enables replay and auditability |
