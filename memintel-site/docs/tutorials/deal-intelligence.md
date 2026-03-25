---
id: deal-intelligence
title: Deal Intelligence for Sales
sidebar_label: Deal Intelligence
---

# Tutorial: Deal Intelligence for Sales

A complete end-to-end walkthrough of building a deterministic deal intelligence system — from raw CRM and email data through to automated sales alerts. This tutorial covers the full architecture, the division of roles and responsibilities, and design guidelines for building a system that stays deterministic in production.

:::note What you'll build
A system that monitors your sales pipeline and automatically alerts reps when deals show signals of risk — with every decision consistent, explainable, and fully reproducible.
:::

---

## The Three Roles

Before anything else, it is worth being explicit about who does what. One of the most common mistakes when building on Memintel is treating it as a single-persona system. It is not. There are three distinct roles, each with a different level of access and a different kind of responsibility.

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Backend / data team | Builds the data pipeline — ingests raw sources, runs signal extraction, delivers typed primitives to the registry |
| **Admin** | Domain expert / platform owner | Registers primitives, configures guardrails, governs the strategy registry and parameter bounds |
| **User** | Sales ops / business user | Expresses intent in plain language — never sees primitives, strategies, or thresholds |

And there is a fourth actor: **Memintel itself** — which compiles user intent into concepts and conditions automatically, within the boundaries the admin has defined.

These roles have a deliberate hierarchy. The data engineer determines what signals are available. The admin determines how those signals can be used. The user determines what they want to monitor. Memintel resolves the rest.

---

## The Architecture

```
Raw Data     →    Signal Extraction    →    Primitives    →    Memintel    →    Decisions
(Data Eng.)       (Data Eng. + LLMs)        (Admin)            (System)         (User)
```

**Layer 1 — Raw Data** *(Data Engineer's responsibility)*

Unstructured, noisy, inconsistent. Never given directly to Memintel.
- Emails, CRM records, Slack messages, call transcripts

**Layer 2 — Signal Extraction** *(Data Engineer's responsibility)*

LLMs and parsers convert raw data into structured signals. This layer is inherently probabilistic — that is fine because its job is interpretation, not decision-making.

*From emails:*
- `response_time_hours` — time from rep's email to customer reply
- `sentiment_score` — 0 to 1, LLM-extracted from email tone
- `last_reply_direction` — customer or rep replied last
- `urgency_detected` — boolean, urgency signals present
- `thread_stalled_days` — days since last reply in thread

*From CRM:*
- `deal_stage` — current stage (categorical)
- `stage_duration_days` — days at current stage
- `deal_value` — contract value
- `last_activity_days` — days since any logged activity

*From Slack:*
- `internal_escalation_flag` — deal escalated internally
- `mention_frequency_7d` — how often deal is referenced

*From calls:*
- `call_completion_rate` — scheduled vs completed ratio
- `next_steps_captured` — next steps recorded after last call

**Layer 3 — Primitives** *(Admin's responsibility)*

Normalised, typed variables registered in Memintel's primitive registry. This is the architecture boundary. Everything before this point is your data pipeline. Everything after this point is deterministic.

:::warning The critical insight
The determinism guarantee only holds from the primitive layer onwards. If your primitives are inconsistently defined, loosely typed, or fed directly from raw LLM outputs without normalisation, Memintel's evaluations become non-deterministic too. Getting primitives right is the most important architectural decision in this system.
:::

---

## Designing Good Primitives

*This section is for the Admin.*

The quality of your primitive design determines the quality of everything above it. A poorly designed primitive layer is the number one reason deterministic evaluation breaks down in practice. Here are the design principles that matter most.

### 1. One signal per primitive

Never bundle two things into one field. `engagement_and_sentiment_combined` is not a primitive — it is a concept. Primitives are atomic. The system composes them; you should not.

```json
// Wrong — two signals in one
{ "name": "engagement_sentiment", "type": "float" }

// Right — separate, composable
{ "name": "sentiment_score",          "type": "float" }
{ "name": "call_completion_rate",     "type": "float" }
```

### 2. Primitives are observable facts, not interpretations

A primitive should represent something you can directly measure or extract. "Deal health" is not a primitive — it is a concept. "Days since last activity" is a primitive.

```json
// Wrong — this is an interpretation, not a fact
{ "name": "deal_health_score", "type": "float" }

// Right — directly measurable facts
{ "name": "last_activity_days",    "type": "int"   }
{ "name": "stage_duration_days",   "type": "int"   }
{ "name": "thread_stalled_days",   "type": "int"   }
```

### 3. Name primitives from the domain, not the system

Names should be immediately legible to a sales ops admin with no engineering background. They are the vocabulary through which business users will express intent.

```json
// Wrong — opaque, system-oriented
{ "name": "email_feature_3",      "type": "float" }
{ "name": "crm_field_last_touch", "type": "int"   }

// Right — domain-legible
{ "name": "sentiment_score",      "type": "float" }
{ "name": "last_activity_days",   "type": "int"   }
```

### 4. Type them strictly — declare nullability explicitly

A `float` that sometimes contains nulls is not a `float` — it is a `float?` (nullable). Using the wrong type causes silent failures at evaluation time. Declare nullability explicitly so the compiler can enforce correct handling.

```json
{ "name": "sentiment_score",       "type": "float"  }  // always present
{ "name": "last_call_sentiment",   "type": "float?" }  // may be null if no calls
{ "name": "deal_stage",            "type": "categorical" }
```

### 5. Distinguish point-in-time from time-series primitives

These are fundamentally different types. A point-in-time primitive is a single value now. A time-series primitive is a sequence of values over a window. Declaring them correctly unlocks different strategies at evaluation time.

```json
// Point-in-time — single value, evaluated now
{ "name": "sentiment_score",       "type": "float"              }

// Time-series — sequence of values, enables z_score and change strategies
{ "name": "sentiment_score_30d",   "type": "time_series<float>" }
{ "name": "activity_count_30d",    "type": "time_series<int>"   }
```

If you want to use a `z_score` or `change` strategy on a signal, you must register it as a `time_series` primitive. The compiler will reject any condition that attempts to apply these strategies to a point-in-time primitive.

### 6. Pair LLM-extracted signals with a confidence score

LLM-extracted signals are probabilistic by nature. Register a confidence score alongside every LLM-derived primitive — this allows the system to weight or gate on confidence when evaluating.

```json
{ "name": "sentiment_score",            "type": "float"  }
{ "name": "sentiment_confidence",       "type": "float"  }
{ "name": "urgency_detected",           "type": "boolean" }
{ "name": "urgency_confidence",         "type": "float"  }
```

### 7. Prefer numeric scores over booleans where possible

A boolean loses the information that a numeric score preserves. `escalation_flag: true` tells you an escalation happened. `escalation_severity: 0.87` tells you how serious it is. The richer primitive enables richer conditions.

```json
// Acceptable — but loses information
{ "name": "internal_escalation_flag",    "type": "boolean" }

// Better — preserves severity for threshold and z_score strategies
{ "name": "internal_escalation_score",  "type": "float"   }
```

Use booleans only when the signal is genuinely binary with no meaningful gradation.

---

## Step 1 — Register Primitives

*Who does this: **Admin**, working with the Data Engineer to agree on the signal catalog.*

With signals extracted and normalised by the data pipeline, the admin registers them in Memintel's primitive registry. This is a governance step — the admin is declaring what signals the system is allowed to use and how they are typed.

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
        "id": "deal.sentiment_confidence",
        "type": "float",
        "source": "email_pipeline",
        "entity": "deal_id",
        "description": "Confidence score for the sentiment extraction, 0-1"
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
        "id": "deal.call_completion_rate",
        "type": "float",
        "source": "calendar_pipeline",
        "entity": "deal_id",
        "description": "Ratio of completed to scheduled calls, 0-1"
    },
    {
        "id": "deal.internal_escalation_score",
        "type": "float",
        "source": "slack_pipeline",
        "entity": "deal_id",
        "description": "Severity of internal escalation signals, 0-1"
    },
]

for p in primitives:
    registry.register_primitive(p)
```

Notice two things. First, `deal.sentiment_score` was produced by an LLM upstream — but once it is registered as a typed `float` primitive, Memintel treats it as a clean numeric input. The probabilistic extraction has already happened. Second, `deal.sentiment_confidence` is registered alongside it, allowing the system to account for extraction reliability.

---

## Step 2 — Register Primitive Resolvers

*Who does this: **Data Engineer**.*

A primitive declaration is just a typed reference — it tells Memintel that a signal called `deal.thread_stalled_days` exists and is an `int`. It contains no database connection, no SQL, no API endpoint. The actual linkage to real data happens through a **primitive resolver** — a function the data team registers that tells Memintel how to fetch that value for a given entity at a given timestamp.

```python
@registry.resolver("deal.thread_stalled_days")
async def resolve_thread_stalled_days(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT DATE_PART('day', $2 - MAX(received_at))::int
        FROM email_messages
        WHERE deal_id = $1
          AND received_at <= $2
    """, entity_id, timestamp)
    return result.scalar()


@registry.resolver("deal.sentiment_score")
async def resolve_sentiment_score(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        SELECT sentiment_score
        FROM email_signal_snapshots
        WHERE deal_id = $1
          AND snapshot_at <= $2
        ORDER BY snapshot_at DESC
        LIMIT 1
    """, entity_id, timestamp)
    return result.scalar()


@registry.resolver("deal.stage_duration_days")
async def resolve_stage_duration_days(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT DATE_PART('day', $2 - stage_entered_at)::int
        FROM deal_stage_history
        WHERE deal_id = $1
          AND stage_entered_at <= $2
        ORDER BY stage_entered_at DESC
        LIMIT 1
    """, entity_id, timestamp)
    return result.scalar()


@registry.resolver("deal.last_activity_days")
async def resolve_last_activity_days(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT DATE_PART('day', $2 - MAX(activity_at))::int
        FROM crm_activity_log
        WHERE deal_id = $1
          AND activity_at <= $2
    """, entity_id, timestamp)
    return result.scalar()
```

Each resolver receives two parameters: `entity_id` (the deal being evaluated) and `timestamp` (the point in time at which to evaluate). What happens inside is entirely up to the data team — SQL queries, Redis lookups, REST API calls, feature store fetches, or anything else.

### Why the timestamp parameter is critical

The timestamp is what makes evaluations reproducible. When Memintel evaluates a condition at a specific timestamp, it passes that timestamp to every resolver in the execution graph. Each resolver is expected to return the value of the primitive **as it was at that exact point in time** — not the current value.

This is what allows you to replay a decision from three months ago and get the exact same result.

```python
# Wrong — ignores timestamp, always returns current state
# Breaks determinism on replay
@registry.resolver("deal.last_activity_days")
async def resolve_last_activity_days(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT DATE_PART('day', NOW() - MAX(activity_at))::int
        FROM crm_activity_log
        WHERE deal_id = $1
    """, entity_id)  # timestamp ignored!
    return result.scalar()

# Right — honours timestamp, returns value as of that moment
@registry.resolver("deal.last_activity_days")
async def resolve_last_activity_days(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT DATE_PART('day', $2 - MAX(activity_at))::int
        FROM crm_activity_log
        WHERE deal_id = $1
          AND activity_at <= $2
    """, entity_id, timestamp)
    return result.scalar()
```

### What data infrastructure supports this

Point-in-time queries require data sources that retain history. Three patterns work well:

| Pattern | How it works | Best for |
|---|---|---|
| **Event / log tables** | Append-only records with timestamps — query `WHERE event_at <= $timestamp` | CRM activity, email events, call logs |
| **Snapshot tables** | Periodic snapshots with a `recorded_at` column — query the latest snapshot before the timestamp | Sentiment scores, computed signals |
| **Feature store** | Systems like Feast or Tecton with built-in point-in-time correctness | Large-scale, multi-team deployments |

If your data team is fetching from a current-state table that gets overwritten — a CRM `deals` table where columns are updated in place — the resolver cannot honour the timestamp. You will need to either add a change-log table or snapshot the relevant fields periodically.

:::warning
If resolvers ignore the timestamp parameter and always return current values, evaluations will produce different results on replay even with the same concept version, condition version, and entity. This breaks the determinism guarantee entirely — not at the Memintel layer, but at the data layer. The data team is responsible for ensuring point-in-time correctness in every resolver.
:::

### The full evaluation chain

When Memintel evaluates `deal.stall_risk` for `deal_acme_corp` at `2024-03-15T09:00:00Z`, here is exactly what happens:

```
1. Compiler looks up execution graph for deal.stall_risk v1.0
2. Graph requires primitives:
     deal.thread_stalled_days
     deal.sentiment_score
     deal.stage_duration_days
     deal.last_activity_days

3. Memintel calls each resolver with (entity_id, timestamp):
     resolve_thread_stalled_days("deal_acme_corp", 2024-03-15T09:00:00Z) → 8
     resolve_sentiment_score("deal_acme_corp",     2024-03-15T09:00:00Z) → 0.29
     resolve_stage_duration_days("deal_acme_corp", 2024-03-15T09:00:00Z) → 34
     resolve_last_activity_days("deal_acme_corp",  2024-03-15T09:00:00Z) → 12

4. Concept computation runs on these four values → stall_risk_score: 0.81
5. Condition evaluates: 0.81 > 0.75 → true
6. Action fires → webhook delivered
```

The resolver layer is the only place where data fetching happens. Everything above it — concept computation and condition evaluation — is a pure function. It receives values and computes results, with no I/O.

---

## Step 3 — Configure Guardrails

*Who does this: **Admin**.*

The guardrails system defines the policy layer that constrains how Memintel resolves user intent. The admin does not write conditions or set thresholds directly. Instead, they define the boundaries within which the system operates — and user intent is resolved deterministically within those boundaries.

Key guardrails for a deal intelligence system:

```python
guardrails = {
    # Which strategies are available for which primitive types
    "type_strategy_map": {
        "int":               ["threshold", "percentile", "change"],
        "float":             ["threshold", "percentile", "z_score", "change"],
        "time_series<float>": ["z_score", "change", "percentile"],
        "boolean":           ["equals"],
        "categorical":       ["equals"],
    },

    # Default threshold parameters per severity level
    # These are what "significantly", "high", "low" resolve to
    "parameter_priors": {
        "sentiment_score": {
            "low_severity":    { "threshold": 0.6 },
            "medium_severity": { "threshold": 0.45 },
            "high_severity":   { "threshold": 0.3 },
        },
        "stage_duration_days": {
            "low_severity":    { "percentile": 60 },
            "medium_severity": { "percentile": 75 },
            "high_severity":   { "percentile": 90 },
        },
        "thread_stalled_days": {
            "low_severity":    { "threshold": 4  },
            "medium_severity": { "threshold": 7  },
            "high_severity":   { "threshold": 12 },
        },
    },

    # How language maps to severity — deterministic lookup, not LLM interpretation
    "bias_rules": {
        "conservative":  "high_severity",
        "early warning": "low_severity",
        "urgent":        "high_severity",
        "monitor":       "low_severity",
    }
}
```

This is important: when a user says *"alert me when a deal is urgently at risk"*, the word "urgently" is not interpreted by an LLM at evaluation time. It maps deterministically to `high_severity` via the bias rules, which then resolves to specific parameter values via `parameter_priors`. The admin defines these mappings. The user benefits from them without ever seeing them.

---

## Step 4 — Memintel Compiles Concepts and Conditions

*Who does this: **Memintel** (automatically).*

This is where the separation of concerns becomes visible. The user expresses intent. Memintel resolves it.

The user does not write concepts. The user does not set thresholds. The user does not select strategies. All of that is resolved by the compiler, working within the guardrails the admin has configured.

```
User intent: "Alert me when a deal is at high risk of stalling"
                              ↓
Guardrails consults primitive registry:
  → thread_stalled_days     (int — threshold strategy applicable)
  → last_activity_days      (int — threshold strategy applicable)
  → sentiment_score         (float — threshold strategy applicable)
  → stage_duration_days     (int — percentile strategy applicable)
                              ↓
"high risk" maps to high_severity via bias rules
                              ↓
Compiler produces:
  Concept:   weighted_sum(thread_pressure, activity_pressure,
                          sentiment_pressure, stage_pressure)
  Condition: stall_risk_score > 0.75  (high_severity threshold)
  Action:    webhook → https://myapp.com/hooks/deal-risk
```

The admin never wrote `0.75`. The user never saw it. The compiler derived it deterministically from the guardrails configuration.

---

## Step 5 — User Creates a Task

*Who does this: **User** (sales ops, business analyst, or sales manager).*

The user interacts with Memintel entirely through natural language. They never configure primitives, write concepts, or set thresholds.

```typescript
import Memintel from '@memintel/sdk';

const client = new Memintel({ apiKey: process.env.MEMINTEL_API_KEY });

// User expresses intent in plain English
const task = await client.tasks.create({
    intent: "Alert me when a deal is at high risk of stalling",
    entityScope: "all_active_deals",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deal-risk"
    },
    dryRun: true  // preview the compiled condition before activating
});

// The user can inspect what the system resolved — but did not author it
console.log(task.condition.strategy);
// {
//   type: "threshold",
//   params: { direction: "above", value: 0.75 },
//   concept_id: "deal.stall_risk"
// }
```

The `dryRun: true` flag is especially useful for users — it lets them review what the system compiled before activating it in production. If the threshold looks wrong, they can adjust their intent wording (e.g. "conservative early warning" instead of "high risk") and preview again.

---

## Step 6 — Evaluate a Deal

*Who does this: **System** (automated, triggered on schedule or event).*

Once tasks are live, evaluation runs automatically. For manual or on-demand evaluation:

```typescript
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

// Signal breakdown — visible to admin and ops for audit
console.log(result.result.explanation.contributions);
// {
//   thread_pressure:   0.35  (thread stalled 8 days — above 7-day threshold)
//   stage_pressure:    0.29  (deal in 83rd percentile for stage duration)
//   sentiment_score:   0.17  (sentiment 0.29 — below 0.3 high-severity cutoff)
// }
```

### Daily pipeline batch

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

## Step 7 — Calibrate Over Time

*Who does this: **User** submits feedback. **Admin** reviews and approves calibration.*

Calibration is a governed process — not an automatic one. The user flags incorrect decisions. The system produces a recommendation. The admin reviews the impact and approves the new version. Nothing changes silently.

```typescript
// User submits feedback on a false positive
await client.feedback.decision({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0",
    entity: "deal_acme_corp_q2",
    timestamp: "2024-03-15T09:00:00Z",
    feedback: "false_positive",
    note: "Deal closed — champion pushed through despite low email activity"
});

// Admin reviews calibration recommendation
const cal = await client.conditions.calibrate({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0",
});

if (cal.status === "recommendation_available") {
    console.log(cal.recommended_params);   // { value: 0.82 } — threshold raised slightly
    console.log(cal.impact.delta_alerts);  // -3 per day

    // Admin approves — creates a new immutable version, never mutates existing
    const applied = await client.conditions.applyCalibration({
        calibrationToken: cal.calibration_token,
    });

    // Admin explicitly rebinds tasks to the new version
    // Nothing changes automatically without this step
    for (const task of applied.tasks_pending_rebind) {
        await client.tasks.update(task.task_id, {
            conditionVersion: applied.new_version,
        });
    }
}
```

---

## Role Summary

| Step | Who | What they do |
|---|---|---|
| Raw data ingestion | **Data Engineer** | Connects email, CRM, Slack, call sources |
| Signal extraction | **Data Engineer** | Runs LLMs and parsers to produce named signals |
| Primitive registration | **Admin** | Registers typed primitives in the registry |
| Guardrails configuration | **Admin** | Defines strategies, parameter priors, bias rules |
| Task creation | **User** | Expresses intent in plain language |
| Intent compilation | **Memintel** | Resolves intent → concepts + conditions |
| Evaluation | **System** | Runs deterministic evaluation on schedule or event |
| Feedback | **User** | Flags false positives and negatives |
| Calibration approval | **Admin** | Reviews and applies threshold adjustments |

---

## The Key Takeaway

The reliability of this system depends entirely on the clarity of its architecture boundaries.

**The Data Engineer** ensures that raw, probabilistic signals are converted into clean typed values before they reach Memintel. The LLM lives here — doing interpretation, which is what it is good at.

**The Admin** ensures that the primitive registry is well-governed and that guardrails constrain intent resolution to sensible, domain-appropriate behaviour.

**The User** expresses what they want in plain language and gets consistent, auditable decisions in return — without ever touching the underlying machinery.

**Memintel** enforces the separation. The LLM cannot execute. The runtime cannot interpret. The compiler cannot be bypassed. That enforced separation is what makes the whole system provably predictable rather than just usually predictable.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Guardrails System](/docs/intro/guardrails) — how admins configure the policy layer
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
- [Common Mistakes](/docs/intro/common-mistakes) — pitfalls to avoid when building on Memintel
