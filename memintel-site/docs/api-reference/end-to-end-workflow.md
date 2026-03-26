---
id: end-to-end-workflow
title: End-to-End Workflow
sidebar_label: End-to-End Workflow
---

# End-to-End Workflow

A complete walkthrough from context configuration to calibration, showing how endpoints chain together and how the guardrails system shapes each step.

---

## Step 0 — Define Application Context (Recommended)

Define domain context before creating primitives or tasks. This gives the LLM the domain knowledge it needs to compile accurate, domain-aware definitions from user intent.

```typescript
// POST /context
const context = await client.context.create({
  domain: {
    description: "B2B SaaS churn detection for mid-market software companies.",
    entities: [
      { name: "user",    description: "individual platform user" },
      { name: "account", description: "company-level subscription" }
    ],
    decisions: ["churn_risk", "expansion_opportunity"]
  },
  behavioural: {
    data_cadence: "batch",
    meaningful_windows: { min: "30d", max: "90d" },
    regulatory: ["GDPR", "SOC2"]
  },
  semantic_hints: [
    { term: "active user", definition: "logged in AND performed core action in last 14 days" },
    { term: "high value account", definition: "ARR above $50,000" }
  ],
  calibration_bias: {
    false_negative_cost: "high",
    false_positive_cost: "medium"
  }
});

// context.version → "v1"
// context.calibration_bias.bias_direction → "recall" (auto-derived)
```

:::tip
Skipping this step is valid — the system works without context. But task definitions will be less domain-accurate and will require more calibration cycles to reach production quality.
:::

---

## Step 1 — Create a Task

```typescript
// Dry run: verify guardrails-resolved strategy and params before committing
const preview = await client.tasks.create({
  intent: "Alert me when churn risk rises significantly",
  entityScope: "user_abc123",
  delivery: { type: "notification" },
  dryRun: true,
});

// "significantly" → high severity → change.high prior → value: 0.10
console.log(preview.condition.strategy);
// { type: "change", params: { direction: "increase", value: 0.10, window: "1d" } }

// Create for real
const task = await client.tasks.create({
  intent: "Alert me when churn risk rises significantly",
  entityScope: "user_abc123",
  delivery: { type: "webhook", endpoint: "https://myapp.com/hooks" },
});
```

---

## Step 2 — Execute and Inspect

```typescript
const pipeline = await client.evaluateFull({
  conceptId: task.concept_id,
  conceptVersion: task.concept_version,
  conditionId: task.condition_id,
  conditionVersion: task.condition_version,
  entity: "user_abc123",
  timestamp: new Date().toISOString(),
  explain: true,
});

console.log(pipeline.result.value);   // 0.87  (Result Rₜ)
console.log(pipeline.decision.value); // true  (Decision Aₜ)
```

---

## Step 3 — Submit Feedback

```typescript
await client.feedback.decision({
  conditionId: task.condition_id,
  conditionVersion: task.condition_version,
  entity: "user_abc123",
  timestamp: "2024-03-15T09:00:00Z",
  feedback: "false_positive",
});
```

---

## Step 4 — Calibrate

```typescript
// feedback_direction derived from stored feedback automatically
const cal = await client.conditions.calibrate({
  conditionId: task.condition_id,
  conditionVersion: task.condition_version,
});
// status: "recommendation_available", recommended_params: { value: 0.15 }
```

---

## Step 5 — Apply and Rebind

```typescript
const applied = await client.conditions.applyCalibration({
  calibrationToken: cal.calibration_token,
});

console.log(applied.new_version); // "1.1"

for (const t of applied.tasks_pending_rebind) {
  await client.tasks.update(t.task_id, { conditionVersion: applied.new_version });
}
```

:::note Lifecycle Complete
The task now runs on condition version `1.1`. Version `1.0` is unchanged — all historical Decisions (Aₜ) made against it remain fully reproducible.
:::
