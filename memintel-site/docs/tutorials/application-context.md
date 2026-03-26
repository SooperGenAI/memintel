---
id: application-context
title: Setting Up Application Context
sidebar_label: Application Context
---

# Tutorial: Setting Up Application Context

Application context is the first thing you should configure before creating primitives or tasks. It gives the LLM the domain knowledge it needs to compile accurate, domain-aware concept and condition definitions from user intent.

This tutorial covers two complete worked examples — SaaS churn detection and fintech fraud detection — and shows concretely how context affects the quality of compiled task definitions and calibration recommendations.

---

## Prerequisites

- Memintel account with admin access
- API key configured

---

## Why Context Matters

Without context, when a user says *"alert me when a high-value account shows churn risk"*, the LLM has no way to know:

- What "high-value" means — is it ARR above $10k, $50k, $500k?
- What time window is meaningful for churn signals — 7 days? 90 days?
- Whether missing a churning account is more costly than a false alarm, or the reverse
- What regulatory frameworks apply — which affects what "significant" means

It compiles something generic. It works — but it will need more calibration cycles before it reaches production accuracy.

With context, the LLM has answers to all of these questions before it starts compiling. The first compiled condition is materially closer to what the user actually needed.

---

## Example 1 — SaaS Churn Detection

### Step 1: Define the context

```bash
POST /context
```

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market software companies. We monitor user engagement and account health to identify accounts at risk of not renewing their subscription.",
    "entities": [
      { "name": "user", "description": "individual platform user within a customer account" },
      { "name": "account", "description": "company-level subscription — the billing and contract unit" }
    ],
    "decisions": ["churn_risk", "expansion_opportunity", "support_escalation"]
  },
  "behavioural": {
    "data_cadence": "batch",
    "meaningful_windows": { "min": "30d", "max": "90d" },
    "regulatory": ["GDPR", "SOC2"]
  },
  "semantic_hints": [
    {
      "term": "active user",
      "definition": "logged in AND performed core action in last 14 days"
    },
    {
      "term": "high value account",
      "definition": "ARR above $50,000"
    },
    {
      "term": "core action",
      "definition": "created, edited, or shared a document — not just a login"
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium"
  }
}
```

**Response:**

```json
{
  "context_id": "ctx_8f3k2m",
  "version": "v1",
  "domain": { ... },
  "behavioural": { ... },
  "semantic_hints": [ ... ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium",
    "bias_direction": "recall"
  },
  "created_at": "2024-03-15T09:00:00Z",
  "is_active": true
}
```

Note the auto-derived `bias_direction: recall` — because `false_negative_cost` (missing a churning account) is higher than `false_positive_cost` (a false alarm). The LLM will bias toward sensitivity when resolving severity language like "at risk" or "significant."

### Step 2: Create a task — with vs without context

**Without context**, a user says: *"Alert me when an account is at risk of churning."*

The compiler has no domain knowledge. It produces something generic:

```json
{
  "concept": "account_activity_score",
  "strategy": "threshold",
  "params": { "value": 0.5 },
  "window": "30d",
  "context_version": null,
  "context_warning": "No active application context exists. Task compiled without domain context — definitions may be less accurate."
}
```

The threshold of 0.5 is a generic midpoint. The window of 30d is a default. Nothing is grounded in the actual domain.

**With context v1 active**, the same user intent compiles to:

```json
{
  "concept": "account_churn_risk",
  "strategy": "composite",
  "components": [
    { "primitive": "account.active_user_rate_30d",   "weight": 0.40 },
    { "primitive": "account.core_action_freq_30d",   "weight": 0.30 },
    { "primitive": "account.seat_utilization_rate",  "weight": 0.20 },
    { "primitive": "account.support_ticket_rate_30d","weight": 0.10 }
  ],
  "params": { "value": 0.38 },
  "window": "30d",
  "context_version": "v1",
  "context_warning": null
}
```

The composite now reflects what "churn risk" actually means in this domain. The 30d window respects `meaningful_windows.min`. The threshold of 0.38 reflects the recall bias — lower than the generic 0.5 midpoint, catching more at-risk accounts at the cost of some additional false positives.

The `semantic_hint` for "active user" is why `account.active_user_rate_30d` was selected — the LLM knew that "active" means "logged in AND performed core action", not just "logged in."

---

## Example 2 — Fintech Fraud Detection

### Step 1: Define the context

```bash
POST /context
```

```json
{
  "domain": {
    "description": "Real-time payment fraud detection for a B2C fintech platform. We monitor individual transactions and customer payment patterns to block fraudulent payments before they settle.",
    "entities": [
      { "name": "customer", "description": "registered platform user with a verified payment method" },
      { "name": "transaction", "description": "individual payment attempt — the primary unit of evaluation" }
    ],
    "decisions": ["fraud_block", "fraud_review", "step_up_auth"]
  },
  "behavioural": {
    "data_cadence": "streaming",
    "meaningful_windows": { "min": "1h", "max": "24h" },
    "regulatory": ["PCI-DSS", "PSD2"]
  },
  "semantic_hints": [
    {
      "term": "unusual velocity",
      "definition": "more than 3 transactions in any 10-minute window"
    },
    {
      "term": "high risk merchant",
      "definition": "MCC code in categories: gambling, cryptocurrency, wire transfer services"
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "medium",
    "false_positive_cost": "high"
  }
}
```

Note the reversed bias here compared to the SaaS example. In fraud detection with automated blocking, a false positive blocks a legitimate customer transaction — which has high cost: customer friction, potential churn, support volume. A false negative allows a fraudulent transaction — which has medium cost because fraud losses are partially recoverable through chargeback processes. The auto-derived `bias_direction` is `precision`.

**Response:**

```json
{
  "context_id": "ctx_9x2k1m",
  "version": "v1",
  "calibration_bias": {
    "false_negative_cost": "medium",
    "false_positive_cost": "high",
    "bias_direction": "precision"
  },
  "created_at": "2024-03-15T10:00:00Z",
  "is_active": true
}
```

### Step 2: See how precision bias affects compilation

A risk analyst says: *"Block transactions that show strong fraud signals."*

With the precision-biased context active, the compiler resolves "strong" to high-severity thresholds — it knows that false positives are costly and sets the bar accordingly:

```json
{
  "concept": "transaction_fraud_risk",
  "strategy": "composite",
  "components": [
    { "primitive": "transaction.velocity_signal",          "weight": 0.35 },
    { "primitive": "transaction.device_fingerprint_risk",  "weight": 0.30 },
    { "primitive": "transaction.merchant_risk_score",      "weight": 0.20 },
    { "primitive": "transaction.amount_vs_baseline_ratio", "weight": 0.15 }
  ],
  "params": { "value": 0.82 },
  "window": "1h",
  "context_version": "v1"
}
```

The threshold of 0.82 is high — reflecting the precision bias. The window of 1h respects `meaningful_windows.min` for a streaming fraud detection context.

The `semantic_hint` for "unusual velocity" is why `transaction.velocity_signal` is the highest-weighted component — the LLM understood that velocity in this domain means a specific pattern (3+ transactions in 10 minutes), not just a generic count.

---

## How Context Affects Calibration

### SaaS churn — recall bias in action

After 6 weeks of operation, the compliance team submits calibration feedback. The statistically optimal threshold based on feedback data alone is 0.78 — but the context has `false_negative_cost: high`:

```json
{
  "condition_id": "churn.high_risk",
  "condition_version": "v3",
  "status": "recommendation_available",
  "statistically_optimal": 0.78,
  "context_adjusted": 0.702,
  "recommended": 0.702,
  "adjustment_explanation": "Threshold adjusted from 0.78 to 0.702 toward recall based on application context (false_negative_cost=high)",
  "calibration_token": "cal_abc123...",
  "feedback_count": 47,
  "false_positive_rate": 0.12,
  "false_negative_rate": 0.08
}
```

The system recommends 0.702 rather than the raw optimal 0.78 — catching more at-risk accounts even at the cost of a slightly higher false positive rate. This is consistent with the declared domain priority: missing a churning account is worse than an unnecessary customer success outreach.

### Fraud detection — precision bias in action

The statistically optimal threshold from feedback is 0.71 — but the context has `false_positive_cost: high`:

```json
{
  "statistically_optimal": 0.71,
  "context_adjusted": 0.781,
  "recommended": 0.781,
  "adjustment_explanation": "Threshold adjusted from 0.71 to 0.781 toward precision based on application context (false_positive_cost=high)"
}
```

The system recommends 0.781 — a higher bar that reduces false blocks of legitimate transactions, consistent with the declared cost structure for an automated blocking action.

### When to use each bias

| Bias direction | Use when | Examples |
|---|---|---|
| **Recall** (lower threshold) | Missing a true signal is more costly than a false alarm | Churn detection, medical alerts, safety monitoring, AML |
| **Precision** (higher threshold) | False alarms have high cost — especially for automated actions | Fraud blocking, automated account suspension, clinical stopping rules |
| **Balanced** | Roughly equal cost on both sides | Internal dashboards, advisory alerts with human review |

---

## Window Clamping

If `behavioural.meaningful_windows` is defined, calibration window parameters are automatically clamped to the declared range.

For the SaaS example with `min: 30d, max: 90d` — if the calibration algorithm recommends a 14-day window based on feedback patterns, the response will note:

```json
{
  "adjustment_explanation": "Window clamped to 30d minimum per application context (meaningful_windows.min=30d). Statistically optimal window was 14d."
}
```

This prevents the calibration engine from recommending windows that are operationally meaningless in the domain — a 14-day churn window in a B2B SaaS context with monthly billing cycles is not actionable.

---

## Checking Context Impact

Use `GET /context/impact` to see how many tasks were compiled under older context versions after you update context:

```bash
GET /context/impact
```

```json
{
  "current_version": "v2",
  "tasks_on_current_version": 14,
  "tasks_on_older_versions": [
    { "version": "v1", "task_count": 8 }
  ],
  "total_stale_tasks": 8
}
```

The 8 tasks still on v1 were compiled before the context update. They continue to run correctly — they are pinned to v1 and fully reproducible. But they do not benefit from the updated domain understanding. Recompile them to incorporate the new context.

---

## Updating Context

When domain understanding changes — a new entity type is introduced, regulatory scope expands, or the cost balance shifts — POST a new context. The previous version is deactivated but retained.

```bash
POST /context
```

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market and enterprise software companies.",
    "entities": [
      { "name": "user",        "description": "individual platform user" },
      { "name": "account",     "description": "company-level subscription" },
      { "name": "workspace",   "description": "team-level organisational unit within an account" }
    ],
    "decisions": ["churn_risk", "expansion_opportunity", "support_escalation", "executive_risk"]
  },
  "behavioural": {
    "data_cadence": "batch",
    "meaningful_windows": { "min": "30d", "max": "90d" },
    "regulatory": ["GDPR", "SOC2", "ISO27001"]
  },
  "semantic_hints": [
    { "term": "active user",      "definition": "logged in AND performed core action in last 14 days" },
    { "term": "high value account","definition": "ARR above $50,000" },
    { "term": "enterprise account","definition": "ARR above $200,000 OR more than 500 seats" }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium"
  }
}
```

The response returns `"version": "v2"` with `"is_active": true`. All new task compilations use v2. Existing tasks remain on v1 until explicitly recompiled.

---

## What You Built

By the end of this tutorial you have:

- Defined a domain-specific application context with entities, semantic hints, regulatory scope, and calibration bias
- Seen how context improves the accuracy of compiled task definitions
- Understood how `bias_direction` affects both compilation thresholds and calibration recommendations
- Used `GET /context/impact` to identify tasks that would benefit from recompilation after a context update

---

## Next Steps

- [Application Context Reference](/docs/intro/application-context) — full API reference for all `/context` endpoints
- [Guardrails System](/docs/intro/guardrails) — how guardrails and context work together at compile time
- [Deal Intelligence Tutorial](/docs/tutorials/deal-intelligence) — see context applied in a sales pipeline use case
