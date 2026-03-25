---
id: deal-intelligence
title: Deal Intelligence for Sales
sidebar_label: Deal Intelligence
---

# Tutorial: Deal Intelligence for Sales

A complete end-to-end walkthrough of building a deterministic deal intelligence system — from raw CRM and email data through to automated sales alerts. This tutorial covers the full architecture, the division of roles and responsibilities, task lifecycle management, and design guidelines for building a system that stays deterministic in production.

:::note What you'll build
A system that monitors your sales pipeline and automatically alerts reps when deals show signals of risk — with every decision consistent, explainable, and fully reproducible.
:::

---

## The Three Roles

Before anything else, it is worth being explicit about who does what. There are three distinct roles, each with a different level of access and a different kind of responsibility.

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Backend / data team | Builds the data pipeline — ingests raw sources, runs signal extraction, writes resolver functions, delivers typed primitives to the registry |
| **Admin** | Domain expert / platform owner | Maintains config files — primitive registry, guardrails — and governs task visibility, version management, and calibration approval |
| **User** | Sales ops / business user | Expresses intent via a bot or UI — creates, views, pauses, and deletes their own tasks |

And a fourth actor: **Memintel itself** — which compiles user intent into concepts and conditions automatically, within the boundaries the admin has defined.

These roles have a deliberate hierarchy. The data engineer determines what signals are available. The admin determines how those signals can be used. The user determines what they want to monitor. Memintel resolves the rest.

---

## The Architecture

```
Raw Data  →  Signal Extraction  →  Primitives  →  Memintel  →  Decisions
(Data Eng.)  (Data Eng. + LLMs)    (Admin cfg)    (System)      (User)
```

**Layer 1 — Raw Data** *(Data Engineer)*

Unstructured, noisy, inconsistent. Never given directly to Memintel.
Emails, CRM records, Slack messages, call transcripts.

**Layer 2 — Signal Extraction** *(Data Engineer)*

LLMs and parsers convert raw data into structured signals. This layer is inherently probabilistic — that is fine because its job is interpretation, not decision-making.

*From emails:* `response_time_hours`, `sentiment_score`, `last_reply_direction`, `urgency_detected`, `thread_stalled_days`

*From CRM:* `deal_stage`, `stage_duration_days`, `deal_value`, `last_activity_days`

*From Slack:* `internal_escalation_score`, `mention_frequency_7d`

*From calls:* `call_completion_rate`, `next_steps_captured`

**Layer 3 — Primitives** *(Admin config + Data Engineer resolvers)*

Normalised, typed variables registered in Memintel's primitive registry. **This is the architecture boundary.** Everything before this point is your data pipeline. Everything after this point is deterministic.

:::warning The critical insight
The determinism guarantee only holds from the primitive layer onwards. If your primitives are inconsistently defined, loosely typed, or fed directly from raw LLM outputs without normalisation, Memintel's evaluations become non-deterministic too. Getting primitives right is the most important architectural decision in this system.
:::

---

## Designing Good Primitives

*This section is for the Admin and Data Engineer.*

### 1. One signal per primitive
Never bundle two things into one field. `engagement_and_sentiment_combined` is not a primitive — it is a concept. Primitives are atomic.

```yaml
# Wrong
- id: deal.engagement_sentiment
  type: float

# Right
- id: deal.sentiment_score
  type: float
- id: deal.call_completion_rate
  type: float
```

### 2. Primitives are observable facts, not interpretations
A primitive should represent something directly measurable. "Deal health" is not a primitive — it is a concept.

```yaml
# Wrong
- id: deal.health_score
  type: float

# Right
- id: deal.last_activity_days
  type: int
- id: deal.stage_duration_days
  type: int
```

### 3. Name primitives from the domain, not the system
Names should be legible to a sales ops admin with no engineering background.

```yaml
# Wrong
- id: email_feature_3
  type: float

# Right
- id: deal.sentiment_score
  type: float
```

### 4. Type them strictly — declare nullability explicitly
A `float` that sometimes contains nulls is `float?`. Wrong types cause silent failures.

```yaml
- id: deal.sentiment_score      # always present
  type: float
- id: deal.last_call_sentiment  # may be null if no calls
  type: float?
```

### 5. Distinguish point-in-time from time-series primitives
Time-series primitives unlock `z_score` and `change` strategies at evaluation time.

```yaml
- id: deal.sentiment_score      # single value now
  type: float
- id: deal.sentiment_score_30d  # sequence over window — enables z_score/change
  type: time_series<float>
```

### 6. Pair LLM-extracted signals with a confidence score

```yaml
- id: deal.sentiment_score
  type: float
- id: deal.sentiment_confidence
  type: float
- id: deal.urgency_detected
  type: boolean
- id: deal.urgency_confidence
  type: float
```

### 7. Prefer numeric scores over booleans where possible
`escalation_severity: 0.87` carries more information than `escalation_flag: true` and enables richer conditions.

---

## Step 1 — Define Primitives (Config File)

*Who does this: **Admin**, working with the Data Engineer to agree on the signal catalog.*

Primitives are defined in a YAML config file maintained by the admin. This file is version-controlled and loaded at system startup. It is a governance artifact — it declares what signals the system is allowed to use and how they are typed. It contains no data fetching logic.

```yaml
# memintel_primitives.yaml

primitives:

  - id: deal.thread_stalled_days
    type: int
    source: email_pipeline
    entity: deal_id
    description: Days since last email reply in the deal thread

  - id: deal.sentiment_score
    type: float
    source: email_pipeline
    entity: deal_id
    description: LLM-extracted sentiment from last 3 customer emails, 0-1

  - id: deal.sentiment_confidence
    type: float
    source: email_pipeline
    entity: deal_id
    description: Confidence score for the sentiment extraction, 0-1

  - id: deal.stage_duration_days
    type: int
    source: crm
    entity: deal_id
    description: Days at current deal stage

  - id: deal.last_activity_days
    type: int
    source: crm
    entity: deal_id
    description: Days since any CRM activity was logged

  - id: deal.call_completion_rate
    type: float
    source: calendar_pipeline
    entity: deal_id
    description: Ratio of completed to scheduled calls, 0-1

  - id: deal.internal_escalation_score
    type: float
    source: slack_pipeline
    entity: deal_id
    description: Severity of internal escalation signals, 0-1
```

This file is loaded at startup:

```python
# application startup
memintel.load_primitives("config/memintel_primitives.yaml")
```

---

## Step 2 — Write Primitive Resolvers (Application Code)

*Who does this: **Data Engineer**.*

A primitive declaration is just a typed reference — it tells Memintel that a signal called `deal.thread_stalled_days` exists and is an `int`. It contains no database connection, no SQL, no API endpoint. The actual linkage to real data happens through **primitive resolvers** — functions the data team writes as part of their application code.

Resolvers live in your application, not in Memintel. Memintel provides the calling convention — it defines when resolvers are called and what they must return. The function body is entirely yours.

```python
# resolvers/deal_resolvers.py — your file, your codebase

from memintel import registry
from datetime import datetime
from db import db  # your database connection

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
```

### Why the timestamp parameter is critical

The timestamp is what makes evaluations reproducible. Every resolver must return the value of the primitive **as it was at that exact point in time** — not the current value.

```python
# Wrong — ignores timestamp, always returns current state
# This breaks determinism on replay
@registry.resolver("deal.last_activity_days")
async def resolve_last_activity_days(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT DATE_PART('day', NOW() - MAX(activity_at))::int
        FROM crm_activity_log
        WHERE deal_id = $1
    """, entity_id)   # timestamp ignored!
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

### Data infrastructure requirements

Point-in-time queries require data sources that retain history:

| Pattern | How it works | Best for |
|---|---|---|
| **Event / log tables** | Append-only records with timestamps | CRM activity, email events, call logs |
| **Snapshot tables** | Periodic snapshots with `recorded_at` column | Sentiment scores, computed signals |
| **Feature store** | Feast, Tecton — built-in point-in-time correctness | Large-scale, multi-team deployments |

:::warning
If resolvers ignore the timestamp parameter, evaluations will produce different results on replay even with identical concept and condition versions. This breaks the determinism guarantee at the data layer — and Memintel cannot detect or compensate for it. The data team is entirely responsible for point-in-time correctness in every resolver.
:::

---

## Step 3 — Configure Guardrails (Config File)

*Who does this: **Admin**.*

Like primitives, guardrails are defined in a YAML config file — not written as application code. This file defines the policy layer that constrains how Memintel resolves user intent. The admin maintains this file; it is loaded at startup.

```yaml
# memintel_guardrails.yaml

type_strategy_map:
  int:                  [threshold, percentile, change]
  float:                [threshold, percentile, z_score, change]
  time_series<float>:   [z_score, change, percentile]
  boolean:              [equals]
  categorical:          [equals]

parameter_priors:
  deal.sentiment_score:
    low_severity:     { threshold: 0.6  }
    medium_severity:  { threshold: 0.45 }
    high_severity:    { threshold: 0.3  }
  deal.thread_stalled_days:
    low_severity:     { threshold: 4  }
    medium_severity:  { threshold: 7  }
    high_severity:    { threshold: 12 }
  deal.stage_duration_days:
    low_severity:     { percentile: 60 }
    medium_severity:  { percentile: 75 }
    high_severity:    { percentile: 90 }

bias_rules:
  conservative:   high_severity
  early warning:  low_severity
  urgent:         high_severity
  monitor:        low_severity

global_default_strategy:   threshold
global_preferred_strategy: percentile
```

When a user says *"alert me when a deal is urgently at risk"*, the word "urgently" maps deterministically to `high_severity` via `bias_rules`, which resolves to specific parameter values via `parameter_priors`. The admin defines these mappings. The user benefits from them without ever seeing them.

```python
# application startup
memintel.load_primitives("config/memintel_primitives.yaml")
memintel.load_guardrails("config/memintel_guardrails.yaml")
```

---

## Step 4 — Memintel Compiles Concepts and Conditions

*Who does this: **Memintel** (automatically).*

The user expresses intent. Memintel resolves it — within the guardrails the admin has configured. The user does not write concepts, set thresholds, or select strategies.

```
User intent: "Alert me when a deal is at high risk of stalling"
                          ↓
Guardrails consults primitive registry:
  → deal.thread_stalled_days   (int — threshold applicable)
  → deal.last_activity_days    (int — threshold applicable)
  → deal.sentiment_score       (float — threshold applicable)
  → deal.stage_duration_days   (int — percentile applicable)
                          ↓
"high risk" → high_severity via bias_rules
                          ↓
Compiler produces:
  Concept:   weighted_sum(thread_pressure, activity_pressure,
                          sentiment_pressure, stage_pressure)
  Condition: stall_risk_score > 0.75
  Action:    webhook → https://myapp.com/hooks/deal-risk
```

The admin never wrote `0.75`. The user never saw it. The compiler derived it deterministically from the guardrails config.

---

## Step 5 — User Creates a Task

*Who does this: **User**, via a bot or UI built by the application team.*

The user never calls the Memintel API directly. In almost every real deployment there is an intermediary — a conversational bot, a configuration UI, or an internal tool — that sits between the user and Memintel. The user interacts with that interface. The interface calls the Memintel API on their behalf.

**Memintel is infrastructure. The interaction layer is the application builder's responsibility.**

### The recommended flow — dryRun → confirm → activate

```
User types in bot / internal tool:
"Alert me when a deal is at high risk of stalling"
          ↓
Bot calls POST /tasks/create with dryRun: true
          ↓
Memintel compiles → returns condition preview
          ↓
Bot shows user in plain English:
"I'll monitor deal stall risk across your active pipeline.
 This will alert you when a deal's stall risk score exceeds 75%.
 Based on last 30 days, this would have fired on 4 deals.
 Shall I activate this?"
          ↓
User confirms
          ↓
Bot calls POST /tasks/create without dryRun → task activated
```

The dryRun step is important — it lets the user review what the system compiled before anything goes live. A well-designed bot translates the compiled condition back into plain English, never exposing raw thresholds or strategy names to the user.

```typescript
// Bot backend code — not user-facing
const preview = await client.tasks.create({
    intent: "Alert me when a deal is at high risk of stalling",
    entityScope: "all_active_deals",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deal-risk"
    },
    dryRun: true
});

// Translate compiled condition back to plain English for the user
const threshold = preview.condition.strategy.params.value;
const recentFirings = preview.estimated_firings_30d;

bot.reply(`I'll alert you when stall risk exceeds ${(threshold * 100).toFixed(0)}%. `
        + `This would have fired on ${recentFirings} deals in the last 30 days. `
        + `Shall I activate this?`);
```

---

## Step 6 — Task Types

*Understanding what kind of task to create.*

Not all tasks work the same way. There are four types, each suited to a different use case.

### Type 1 — One-off evaluation

A single immediate evaluation. The user asks a question and gets an answer right now. No ongoing monitoring. No alerts.

```
User: "What is the current risk score for the Acme Corp deal?"
Bot:  Calls POST /evaluate/full for deal_acme_corp
      Returns: "Acme Corp stall risk is 81%. Top signal: thread
               stalled for 8 days. No action has been triggered."
```

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
// Immediate response — no task created, no ongoing monitoring
```

### Type 2 — Ongoing monitoring task

A persistent task that runs on a schedule or event trigger. Fires an alert whenever the condition is met. This is the primary use case for Memintel in production.

```
User: "Alert me whenever a deal is at high risk of stalling"
Bot:  Creates task → task_id: tsk_8f3k2
      "Done. I'll check your active pipeline every hour and
       alert you here when a deal's stall risk crosses 75%."
```

```typescript
const task = await client.tasks.create({
    intent: "Alert me whenever a deal is at high risk of stalling",
    entityScope: "all_active_deals",
    schedule: { frequency: "hourly" },
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/deal-risk" }
});
// Returns task_id — task now runs on schedule automatically
```

### Type 3 — Batch evaluation

A one-time evaluation across many entities at once. Returns a ranked report. No ongoing monitoring.

```
User: "Give me a risk report across all Q2 deals right now"
Bot:  Evaluates all 47 active Q2 deals
      Returns ranked report: top 5 at-risk deals with scores and drivers
```

```typescript
const activeDeals = await crm.getActiveDeals({ pipeline: "Q2" });

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

const ranked = results
    .sort((a, b) => b.result.value - a.result.value)
    .slice(0, 5);
```

### Type 4 — Historical replay

A point-in-time evaluation. What was the state of a deal at a specific moment in the past? Fully deterministic — same result every time.

```
User: "What was the stall risk for Acme Corp on March 15th?"
Bot:  Evaluates deal_acme_corp at 2024-03-15T09:00:00Z
      Returns: "On March 15th, Acme Corp's stall risk was 81%.
               Thread had been stalled 8 days at that point."
```

```typescript
const result = await client.evaluateFull({
    concept_id: "deal.stall_risk",
    concept_version: "1.0",
    condition_id: "deal.at_risk_of_stalling",
    condition_version: "1.0",
    entity: "deal_acme_corp_q2",
    timestamp: "2024-03-15T09:00:00Z",  // specific past timestamp
    explain: true
});
```

---

## Step 7 — The System Response Loop

*What happens after an ongoing task is activated.*

When a user activates an ongoing monitoring task, the request-response interaction ends and a new loop begins. Understanding this loop is critical for designing alert delivery and user experience.

```
Task activated
      ↓
Schedule fires (e.g. every hour)
      ↓
For each entity in scope:
  Memintel calls resolvers with (entity_id, timestamp)
  Resolvers fetch point-in-time values from data sources
  Concept computation runs on primitive values
  Condition evaluates: does score cross threshold?
      ↓
  If YES → Action triggered
              ↓
           Webhook called / notification sent
           Decision logged with full audit trail
           Alert delivered to user via configured channel
      ↓
  If NO  → No action. Decision logged.
      ↓
Next schedule tick
```

### What the user receives

When a condition fires, the user receives an alert via their configured channel — Slack, email, webhook, or in-app notification. A well-designed alert includes:

```
🚨 Deal at risk: Acme Corp (Enterprise Q2)

Stall risk score: 81%  (threshold: 75%)

Top signals:
  • Email thread stalled for 8 days
  • Deal in negotiation stage for 34 days (83rd percentile)
  • Sentiment score: 0.29 (below 0.30 threshold)

→ View deal   → Mark as reviewed   → Adjust alert sensitivity
```

The "Adjust alert sensitivity" action feeds directly into the feedback and calibration cycle.

### What gets logged

Every evaluation — whether or not the condition fired — is logged with its full audit trail:

```json
{
  "decision_id":       "dec_9x2k1m",
  "task_id":           "tsk_8f3k2",
  "entity":            "deal_acme_corp_q2",
  "timestamp":         "2024-03-15T09:00:00Z",
  "concept_id":        "deal.stall_risk",
  "concept_version":   "1.0",
  "condition_id":      "deal.at_risk_of_stalling",
  "condition_version": "1.0",
  "result_value":      0.81,
  "decision_value":    true,
  "action_triggered":  true,
  "contributions": {
    "thread_pressure":  0.35,
    "stage_pressure":   0.29,
    "sentiment_score":  0.17
  }
}
```

This log is immutable. Any decision can be replayed at any time by re-evaluating with the same concept version, condition version, entity, and timestamp.

---

## Step 8 — Task Management

### User controls

*Users can only see and manage their own tasks.*

**View task status:**

```typescript
// List all tasks created by this user
const tasks = await client.tasks.list({ owner: "current_user" });

tasks.forEach(task => {
    console.log(`${task.id}: ${task.status}`);
    // status: active | paused | suspended | archived
    console.log(`  Last fired: ${task.last_fired_at}`);
    console.log(`  Next run:   ${task.next_run_at}`);
    console.log(`  Fired ${task.fire_count_30d} times in last 30 days`);
});
```

**Pause and resume:**

```typescript
// Pause — stops evaluation, no alerts fired
await client.tasks.pause("tsk_8f3k2");

// Resume — evaluation restarts on next schedule tick
await client.tasks.resume("tsk_8f3k2");
```

**Delete:**

```typescript
// Permanently removes the task
// Historical decisions and audit log are retained
await client.tasks.delete("tsk_8f3k2");
```

**Rebind to a new condition version:**

When a condition is calibrated and a new version is created, the user is notified and prompted to rebind. The task remains on the old version until explicitly rebound — nothing changes automatically.

```typescript
// User is notified: "A new version of your stall risk condition
// is available. Threshold updated from 0.75 to 0.82 (-3 alerts/day).
// Update your task?"

await client.tasks.update("tsk_8f3k2", {
    conditionVersion: "1.1"  // explicit rebind — user's decision
});
```

### Admin visibility and controls

*Admins can see and manage all tasks across all users.*

**View all tasks:**

```typescript
// Admin view — all tasks, all users
const allTasks = await adminClient.tasks.list();

// Filter by condition version — useful for version migration
const oldVersionTasks = await adminClient.tasks.list({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0"
});

console.log(`${oldVersionTasks.length} tasks still on v1.0`);
```

**Task metrics dashboard:**

```typescript
const metrics = await adminClient.tasks.metrics({
    conditionId: "deal.at_risk_of_stalling"
});

console.log(metrics.active_task_count);       // 23
console.log(metrics.firing_rate_per_day);     // 4.2
console.log(metrics.false_positive_rate);     // 0.18
console.log(metrics.version_distribution);
// { "1.0": 15, "1.1": 8 }  — how many tasks on each version
```

**Suspend a task:**

```typescript
// Admin can suspend any task — e.g. if a data source goes down
await adminClient.tasks.suspend("tsk_8f3k2", {
    reason: "email_pipeline outage — primitives unreliable"
});
// User is notified their task has been suspended and why
```

**Deprecate a condition version:**

```typescript
// Marks a version as deprecated
// All tasks on this version are suspended
// Users are notified and prompted to rebind
await adminClient.conditions.deprecateVersion({
    conditionId: "deal.at_risk_of_stalling",
    version: "1.0",
    message: "Threshold recalibrated — please rebind to v1.1"
});
```

---

## Step 9 — The Feedback and Calibration Cycle

*This is a continuous loop, not a one-time step.*

Calibration is a governed process. The user flags incorrect decisions. The system accumulates feedback and produces a recommendation. The admin reviews the impact and approves the new version. Tasks are explicitly rebound. Nothing changes silently.

```
User receives alert
      ↓
User marks alert as correct / false positive / false negative
      ↓
Feedback accumulated in condition's feedback store
      ↓
(When enough feedback exists)
Admin requests calibration recommendation
      ↓
System recommends adjusted parameters + estimated impact
      ↓
Admin reviews: "Raise threshold from 0.75 to 0.82 → -3 alerts/day"
      ↓
Admin approves → new condition version created (immutable)
      ↓
Users notified: "New version available for your task"
      ↓
Users explicitly rebind their tasks to new version
      ↓
Loop continues on next schedule tick with new parameters
```

```typescript
// Step 1 — User submits feedback
await client.feedback.decision({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0",
    entity: "deal_acme_corp_q2",
    timestamp: "2024-03-15T09:00:00Z",
    feedback: "false_positive",
    note: "Deal closed — champion pushed through despite low activity"
});

// Step 2 — Admin requests calibration recommendation
const cal = await adminClient.conditions.calibrate({
    conditionId: "deal.at_risk_of_stalling",
    conditionVersion: "1.0",
});

// Step 3 — Admin reviews impact
console.log(cal.recommended_params);    // { value: 0.82 }
console.log(cal.impact.delta_alerts);   // -3 per day
console.log(cal.feedback_summary);
// { false_positives: 12, false_negatives: 2, correct: 31 }

// Step 4 — Admin approves → new immutable version created
const applied = await adminClient.conditions.applyCalibration({
    calibrationToken: cal.calibration_token,
});

console.log(applied.new_version);  // "1.1"
console.log(applied.tasks_pending_rebind.length);  // 23 tasks need rebinding

// Step 5 — Users are notified and rebind explicitly
// (Each user does this themselves — nothing is automatic)
await client.tasks.update("tsk_8f3k2", {
    conditionVersion: applied.new_version,
});
```

---

## Full Lifecycle Diagram

```
SETUP (one time)
────────────────────────────────────────────────────────
Data Engineer:  Write resolver functions (application code)
Admin:          Define primitives → memintel_primitives.yaml
Admin:          Define guardrails  → memintel_guardrails.yaml
System:         Load config at startup


TASK CREATION
────────────────────────────────────────────────────────
User:           Express intent via bot / UI
Bot:            POST /tasks/create?dryRun=true
Memintel:       Compile intent → concept + condition
Bot:            Show plain-English preview to user
User:           Confirm
Bot:            POST /tasks/create → task activated


ONGOING EVALUATION LOOP
────────────────────────────────────────────────────────
Schedule tick
  → Resolvers fetch point-in-time primitive values
  → Concept computation runs
  → Condition evaluates
  → If fired: action triggered, alert delivered, decision logged
  → If not:   decision logged, no action


TASK MANAGEMENT
────────────────────────────────────────────────────────
User:   View / pause / resume / delete own tasks
Admin:  View all tasks, metrics, version distribution
Admin:  Suspend tasks, deprecate condition versions


CALIBRATION CYCLE
────────────────────────────────────────────────────────
User:   Submit feedback on alerts (correct / FP / FN)
Admin:  Request calibration recommendation
Admin:  Review impact (delta alerts/day)
Admin:  Approve → new condition version created
User:   Rebind task to new version (explicit, never automatic)
        ↓
Back to ONGOING EVALUATION LOOP with new parameters
```

---

## Role Summary

| Step | Who | What |
|---|---|---|
| Write resolvers | **Data Engineer** | Application code — fetches primitive values point-in-time |
| Define primitives | **Admin** | `memintel_primitives.yaml` — loaded at startup |
| Configure guardrails | **Admin** | `memintel_guardrails.yaml` — loaded at startup |
| Compile intent | **Memintel** | Auto-resolves concepts + conditions from user intent |
| Create task | **User** | Via bot or UI — never directly via API |
| Ongoing evaluation | **System** | Scheduled, deterministic, fully logged |
| View own tasks | **User** | Status, last fired, next run, alert history |
| View all tasks | **Admin** | Metrics, version distribution, firing rates |
| Submit feedback | **User** | Flags false positives / negatives |
| Approve calibration | **Admin** | Reviews impact, creates new version |
| Rebind task | **User** | Explicit — nothing changes automatically |

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Guardrails System](/docs/intro/guardrails) — how admins configure the policy layer
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
- [Common Mistakes](/docs/intro/common-mistakes) — pitfalls to avoid when building on Memintel
