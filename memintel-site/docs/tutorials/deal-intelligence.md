---
id: deal-intelligence
title: Deal Intelligence for Sales
sidebar_label: Deal Intelligence
---

# Tutorial: Deal Intelligence for Sales

A complete end-to-end walkthrough of building a deterministic deal intelligence system — from raw CRM and email data through to automated sales alerts. This tutorial covers the full architecture, including where Memintel sits and what comes before it.

:::note What you'll build
A system that monitors your sales pipeline and automatically alerts reps when deals show signals of risk — with every decision consistent, explainable, and fully reproducible.
:::

---

## The Right Mental Model

Before writing a single line of code, it is worth understanding the three-layer architecture that makes this system work. Getting this right is what separates a reliable production system from one that collapses back into probabilistic LLM reasoning.

```
Layer 1          Layer 2               Layer 3
──────────       ──────────────────    ──────────────────────────
Raw Data    →    Signal Extraction →   Memintel
(messy,          (LLMs + parsers,      (typed primitives →
 unstructured)    semi-structured)      deterministic decisions)
```

### Layer 1 — Data Sources

This is your raw data. Unstructured, noisy, inconsistent. You do not give this directly to Memintel.

- **Emails** — threads, replies, response times, tone
- **CRM** — deal stages, activity logs, close dates, deal values
- **Slack** — internal discussions, escalation mentions, deal references
- **Calls** — transcripts, duration, talk ratios, next steps captured

### Layer 2 — Signal Extraction

This is where LLMs and parsers do their job: converting raw unstructured data into structured, named signals. This layer is inherently probabilistic — and that is fine, because its job is interpretation, not decision-making.

*From emails:*
- `response_time_hours` — how long the customer took to reply
- `sentiment_score` — 0 to 1, extracted by LLM from email tone
- `last_reply_direction` — did the customer reply last, or the rep?
- `urgency_detected` — boolean, did the email contain urgency signals?
- `thread_stalled_days` — days since last email in the thread

*From CRM:*
- `deal_stage` — current stage (categorical: prospecting, negotiation, etc.)
- `stage_duration_days` — how many days at the current stage
- `deal_value` — contract value
- `last_activity_days` — days since any CRM activity was logged

*From Slack:*
- `internal_escalation_flag` — boolean, has this deal been escalated internally?
- `mentions_of_deal` — how often the deal is referenced in the last 7 days

*From calls:*
- `call_completion_rate` — scheduled vs completed calls
- `next_steps_captured` — boolean, were next steps recorded after the last call?

### Layer 3 — Primitives (where Memintel begins)

Once signals are extracted, they are normalised into typed primitives and handed to Memintel. **This is the architecture boundary.** Everything before this point is your data pipeline. Everything after this point is deterministic.

```json
{ "name": "thread_stalled_days",       "type": "int"         }
{ "name": "customer_sentiment_score",  "type": "float"       }
{ "name": "deal_stage",                "type": "categorical" }
{ "name": "stage_duration_days",       "type": "int"         }
{ "name": "last_activity_days",        "type": "int"         }
{ "name": "internal_escalation_flag",  "type": "boolean"     }
```

:::warning The critical insight
If your primitives are wrong — inconsistently defined, loosely typed, or directly fed from raw LLM outputs without normalisation — Memintel's evaluations become non-deterministic too. The determinism guarantee only holds from the primitive layer onwards. Getting primitives right is the most important architectural decision in this system.
:::

---

## Who Defines What

| Layer | Who defines it | Nature |
|---|---|---|
| Data sources | System / integrations | Raw, messy |
| Signal extraction | System (LLM + parsers + ETL) | Semi-structured |
| Primitives | System (standardised registry) | Clean, typed |
| Intent | Admin | High-level natural language |
| Concepts + Conditions | Memintel compiler | Derived from intent + guardrails |
| Actions | Admin | Configured endpoints |

The admin does **not** define primitives manually from scratch. Instead they express intent in plain language — *"detect deal risk based on customer engagement and sentiment"* — and Memintel's guardrails system, working from the registered primitive catalog, resolves which primitives are relevant and how to combine them.

This is the intent-driven model in practice:

```
Admin: "Flag deals that are likely to stall"
         ↓
Guardrails consults primitive registry:
  → thread_stalled_days
  → customer_sentiment_score
  → last_activity_days
  → stage_duration_days
         ↓
Compiler produces:
  IF thread_stalled_days > X
  AND sentiment_score < Y
  AND stage_duration > Z
  THEN trigger: deal_at_risk
```

The admin never manually specifies thresholds or writes logic. They express intent. Memintel resolves the rest — deterministically, within the constraints the guardrails system defines.

---

## Step 1 — Register Your Primitives

With signals extracted and normalised by your data pipeline, register them in Memintel's primitive registry:

```python
primitives = [
    {
        "id": "deal.thread_stalled_days",
        "type": "int",
        "source": "email_pipeline",
        "entity": "deal_id",
        "description": "Days since last email reply in the deal thread"
    },
    {
        "id": "deal.sentiment_score",
        "type": "float",
        "source": "email_pipeline",
        "entity": "deal_id",
        "description": "LLM-extracted sentiment from last 3 customer emails, 0-1"
    },
    {
        "id": "deal.stage_duration_days",
        "type": "int",
        "source": "crm",
        "entity": "deal_id",
        "description": "Days at current deal stage"
    },
    {
        "id": "deal.last_activity_days",
        "type": "int",
        "source": "crm",
        "entity": "deal_id",
        "description": "Days since any CRM activity was logged"
    },
    {
        "id": "deal.internal_escalation_flag",
        "type": "boolean",
        "source": "slack_pipeline",
        "entity": "deal_id",
        "description": "True if deal has been escalated internally in last 7 days"
    },
    {
        "id": "deal.call_completion_rate",
        "type": "float",
        "source": "calendar_pipeline",
        "entity": "deal_id",
        "description": "Ratio of completed to scheduled calls"
    },
]

for p in primitives:
    registry.register_primitive(p)
```

Notice that `deal.sentiment_score` was produced by an LLM upstream — but once it is registered as a typed `float` primitive, Memintel treats it as a clean numeric input. The probabilistic extraction has already happened. Everything from here is deterministic.

---

## Step 2 — Build Your Concepts

Concepts combine primitives into meaningful signals. They answer: *"what is the current state of this deal?"*

### Engagement Score

Combines communication and activity signals into a single 0-1 score.

```python
define_concept({
    "id": "deal.engagement_score",
    "inputs": [
        "deal.sentiment_score",
        "deal.call_completion_rate",
        "deal.thread_stalled_days",
        "deal.last_activity_days"
    ],
    "features": {
        "comms_health": {
            "op": "weighted_sum",
            "inputs": ["deal.sentiment_score", "deal.call_completion_rate"],
            "weights": [0.5, 0.5]
        },
        "activity_recency": {
            "op": "inverse_decay",
            "input": "deal.last_activity_days",
            "halflife": 7
        }
    },
    "compute": "weighted_sum(comms_health, activity_recency, weights=[0.6, 0.4])"
})
```

### Stall Risk Score

Combines signals that indicate a deal is stalling — not just inactive, but actively deteriorating.

```python
define_concept({
    "id": "deal.stall_risk",
    "inputs": [
        "deal.thread_stalled_days",
        "deal.stage_duration_days",
        "deal.internal_escalation_flag",
        "deal.engagement_score"
    ],
    "features": {
        "thread_pressure": {
            "op": "threshold_step",
            "input": "deal.thread_stalled_days",
            "cutoff": 5,
            "above_value": 1.0,
            "below_value": 0.0
        },
        "stage_pressure": {
            "op": "percentile_rank",
            "input": "deal.stage_duration_days"
        }
    },
    "compute": "weighted_sum(thread_pressure, stage_pressure, 1 - engagement_score, weights=[0.35, 0.35, 0.30])"
})
```

---

## Step 3 — Define Your Conditions

Conditions evaluate concept outputs and produce decisions. They answer: *"does this deal's current state warrant action?"*

### Condition 1 — Deal at Risk of Stalling

```typescript
const stallingCondition = await client.tasks.create({
    intent: "Alert me when a deal is at high risk of stalling",
    entityScope: "all_active_deals",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deal-stalling"
    },
    dryRun: true  // preview the compiled condition before activating
});

// See exactly how "high risk" was resolved
console.log(stallingCondition.condition.strategy);
// {
//   type: "threshold",
//   params: { direction: "above", value: 0.75 },
//   concept_id: "deal.stall_risk"
// }
```

### Condition 2 — Deal Going Cold

Uses a `change` strategy to catch deals that were healthy but are deteriorating.

```typescript
const goingColdCondition = await client.tasks.create({
    intent: "Alert me when deal engagement drops significantly week over week",
    entityScope: "all_active_deals",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deal-cold"
    }
});

// "significantly week over week" resolves to:
// { type: "change", params: { direction: "decrease", value: 0.25, window: "7d" } }
```

### Condition 3 — Internally Escalated Deal

Uses `equals` strategy on a boolean primitive — no concept needed for simple flags.

```typescript
const escalationCondition = await client.tasks.create({
    intent: "Alert me immediately when a deal is internally escalated",
    entityScope: "all_active_deals",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deal-escalated"
    }
});

// Directly evaluates the boolean primitive
// { type: "equals", params: { value: true } }
```

---

## Step 4 — Evaluate a Deal

With everything wired up, evaluating a specific deal is a single API call:

```typescript
import Memintel from '@memintel/sdk';

const client = new Memintel({ apiKey: process.env.MEMINTEL_API_KEY });

const result = await client.evaluateFull({
    concept_id: "deal.stall_risk",
    concept_version: "1.0",
    condition_id: "deal.at_risk_of_stalling",
    condition_version: "1.0",
    entity: "deal_acme_corp_q2",
    timestamp: new Date().toISOString(),
    explain: true
});

console.log(result.result.value);    // 0.81 — stall risk score
console.log(result.decision.value);  // true — threshold crossed

// Full signal breakdown
console.log(result.result.explanation.contributions);
// {
//   thread_pressure:   0.35  (thread stalled 8 days — above 5-day cutoff)
//   stage_pressure:    0.29  (deal in 78th percentile for stage duration)
//   engagement_score:  0.17  (sentiment 0.31, no calls completed)
// }
```

The `explain: true` output is what makes this genuinely useful for sales ops. Your rep does not just get an alert — they get a breakdown: the email thread has been stalled for 8 days, the deal has been in negotiation longer than 78% of similar deals, and engagement signals are low.

---

## Step 5 — Run Your Daily Pipeline Review

Evaluate every active deal in one batch:

```typescript
const activeDeals = await crm.getActiveDeals();

const results = await Promise.all(
    activeDeals.map(deal =>
        client.evaluateFull({
            concept_id: "deal.stall_risk",
            concept_version: "1.0",
            condition_id: "deal.at_risk_of_stalling",
            condition_version: "1.0",
            entity: deal.id,
            timestamp: new Date().toISOString(),
            explain: true,
        })
    )
);

// Sort by risk score and surface the top deals
const atRisk = results
    .filter(r => r.decision.value === true)
    .sort((a, b) => b.result.value - a.result.value)
    .slice(0, 5);

atRisk.forEach(r => {
    const topDriver = Object.entries(r.result.explanation.contributions)
        .sort(([,a], [,b]) => (b as number) - (a as number))[0];

    console.log(`${r.decision.entity}: ${(r.result.value * 100).toFixed(0)}% risk`);
    console.log(`  Top signal: ${topDriver[0]}`);
});
```

---

## Step 6 — Calibrate Over Time

After a few weeks you will see false positives — deals flagged as at-risk that closed successfully. Feed this back to tighten the thresholds:

```typescript
// Mark a false positive
await client.feedback.decision({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0",
    entity: "deal_acme_corp_q2",
    timestamp: "2024-03-15T09:00:00Z",
    feedback: "false_positive",
    note: "Deal closed — champion pushed through despite low email activity"
});

// Get calibration recommendation
const cal = await client.conditions.calibrate({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0",
});

if (cal.status === "recommendation_available") {
    console.log(cal.recommended_params);   // { value: 0.82 }
    console.log(cal.impact.delta_alerts);  // -3 per day

    // Apply — creates a new immutable version, never mutates existing
    const applied = await client.conditions.applyCalibration({
        calibrationToken: cal.calibration_token,
    });

    // Explicitly rebind tasks to the new version
    for (const task of applied.tasks_pending_rebind) {
        await client.tasks.update(task.task_id, {
            conditionVersion: applied.new_version,
        });
    }
}
```

---

## What You've Built

| Capability | How it works |
|---|---|
| **Consistent scoring** | Same primitive values always produce the same risk score |
| **Explainable alerts** | Every alert shows exactly which signals drove the decision |
| **Auditable history** | Any past decision can be replayed with the original primitive values |
| **Controlled calibration** | Thresholds evolve via structured feedback — never silently |
| **Architecture clarity** | LLMs do signal extraction; Memintel does evaluation — clean separation |

---

## The Key Takeaway

The reliability of this system comes entirely from the architecture boundary.

**LLMs sit upstream** — in signal extraction, converting raw emails and CRM data into typed signals. They are allowed to be probabilistic here because their job is interpretation.

**Memintel sits downstream** — evaluating clean typed primitives with deterministic logic. Once a `sentiment_score: 0.31` primitive is registered, the evaluation engine never calls an LLM again. The same score at the same timestamp always produces the same decision.

Get the primitive layer right, and Memintel becomes a true deterministic evaluation engine. Get it wrong — by feeding raw LLM outputs directly into conditions without normalisation — and you lose the determinism guarantee entirely.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Guardrails System](/docs/intro/guardrails) — how admins constrain intent resolution
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
- [Common Mistakes](/docs/intro/common-mistakes) — pitfalls to avoid when building on Memintel
