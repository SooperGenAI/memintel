---
id: conditions
title: Conditions
sidebar_label: Conditions
---

# Conditions

Conditions are the significance layer (φ). They define when a concept's Result (Rₜ) is meaningful enough to act on. Conditions are immutable once created — changes produce new versions.

| Endpoint | Level | Description |
|---|---|---|
| `POST /conditions/explain` | **Definition-level** | Explains the condition logic — strategy, parameters, and why they were selected. No entity or execution required. |
| `POST /decisions/explain` | **Instance-level** | Explains a specific Decision (Aₜ) for a given entity at a given timestamp — why it fired, which signals drove the Result (Rₜ). |

---

## Get Condition

```
GET /conditions/{id}
```

Retrieves a condition definition. Both `id` (path) and `version` (query) are required — no implicit latest resolution.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `id` | string | Required (path) | Fully qualified condition id (`namespace.id`). Example: `org.high_churn`. |
| `version` | string | Required (query) | Explicit condition version. Example: `"1.0"`. |

### Response — ConditionDefinition

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_id` / `version` | string | Always | Identifier and version. |
| `concept_id` / `concept_version` | string | Always | Pinned concept this condition evaluates. |
| `strategy` | StrategyDefinition | Always | Strategy type and resolved parameters. |
| `deprecated` | boolean | Always | Whether this version is deprecated. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Condition definition. |
| **401** | Unauthorised. |
| **404** | Not found. |

### TypeScript Example

```typescript
const condition = await client.conditions.get({
  conditionId: "org.high_churn",
  version: "1.0",
});
console.log(condition.strategy);
// { type: "threshold", params: { direction: "above", value: 0.8 } }
```

---

## Explain Condition

```
POST /conditions/explain
```

**Definition-level.** Explains the condition logic itself — what strategy is applied, why those parameters were selected, and the condition's relationship to its concept. Use for audit trails, compliance documentation, and reviewing guardrails decisions.

:::note Determinism Guarantee
Same `condition_id`, `condition_version`, and `timestamp` always return the same explanation. Application context may influence terminology in natural language summaries — never decision values, parameter values, or attribution weights.
:::

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_id` | string | **Required** | Fully qualified condition identifier. |
| `condition_version` | string | **Required** | Explicit version. No implicit latest. |
| `timestamp` | datetime | Optional | ISO 8601 UTC. Ensures reproducible explanation. |

### Response — ConditionExplanation

| Parameter | Type | Required | Description |
|---|---|---|---|
| `natural_language_summary` | string | Always | Human-readable description of what the condition evaluates and when it fires. |
| `parameter_rationale` | string | Always | Why these parameter values were selected — from guardrails priors, bias rules, or explicit user input. |
| `strategy` | object | Always | Strategy type and resolved parameters. |
| `concept_id` | string | Always | The concept this condition evaluates. |

### Response Codes

| Status | Description |
|---|---|
| **200** | ConditionExplanation. |
| **401** | Unauthorised. |
| **404** | Condition not found. |

### TypeScript Example

```typescript
const exp = await client.conditions.explain({
  conditionId: "org.high_churn",
  conditionVersion: "1.0",
});

console.log(exp.natural_language_summary);
// "Fires when churn_risk score exceeds 0.8 — the high-severity threshold
//  based on guardrails priors."

console.log(exp.parameter_rationale);
// "Threshold 0.8 from primitive-level prior at high severity.
//  Severity resolved from \"significantly\" via severity vocabulary."
```

---

## Calibrate Condition

```
POST /conditions/calibrate
```

Analyses stored feedback and/or a target alert volume to recommend adjusted parameters. Does **not** modify the existing condition — call `POST /conditions/apply-calibration` to create a new version.

**Strategy coverage:** Supports `threshold`, `percentile`, `change`, and `z_score`. For `equals` strategy, always returns `status: no_recommendation` with `reason: not_applicable_strategy` — no numeric parameter to adjust.

:::note Bounds Enforcement
Recommendations are validated against guardrails `threshold_bounds`. If exceeded, `on_bounds_exceeded` applies (`clamp` or `reject`). Bounds cannot be crossed.
:::

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_id` | string | **Required** | Fully qualified condition identifier. |
| `condition_version` | string | **Required** | Version to calibrate. |
| `target.alerts_per_day` | number | Optional | Target alert frequency. |
| `feedback_direction` | enum | Optional | `tighten` \| `relax`. Overrides direction derived from stored feedback. |

### Response — CalibrationResult

| Parameter | Type | Required | Description |
|---|---|---|---|
| `status` | enum | Always | `recommendation_available` \| `no_recommendation`. |
| `recommended_params` | object | Conditional | Strategy parameter key-values when recommendation available. |
| `calibration_token` | string | Conditional | Pass to `apply-calibration`. |
| `impact.delta_alerts` | number | Conditional | Estimated change in daily alert volume. |
| `current_params` | object | Always | Current parameters for comparison. |
| `no_recommendation_reason` | string | Conditional | `bounds_exceeded` \| `not_applicable_strategy` \| `insufficient_data`. |
| `statistically_optimal` | float | Conditional | The raw statistically optimal parameter value based on feedback data alone, before any context bias adjustment. |
| `context_adjusted` | float \| null | Conditional | The bias-adjusted value after applying `calibration_bias` from application context. `null` if no context or no `calibration_bias` defined. |
| `recommended` | float | Conditional | The final recommended value. Equals `context_adjusted` if bias was applied, otherwise equals `statistically_optimal`. |
| `adjustment_explanation` | string \| null | Conditional | Human-readable explanation of any bias adjustment applied. `null` if no adjustment was made. |

**Bias adjustment logic:**

When application context defines `calibration_bias`, the statistically optimal threshold is adjusted before being returned as `recommended`:

- `bias_direction = recall` (false_negative_cost > false_positive_cost) → threshold lowered. Adjustment: `high` = 10%, `medium` = 5%, `low` = 2%
- `bias_direction = precision` (false_positive_cost > false_negative_cost) → threshold raised. Same adjustment factors apply
- `bias_direction = balanced` → no adjustment. `recommended = statistically_optimal`
- Adjusted values are always clamped to `[0.0, 1.0]`
- If `behavioural.meaningful_windows` is defined in context, any window parameter in the recommendation is clamped to the declared min/max range. The `adjustment_explanation` field notes when clamping is applied

**Example response with context-adjusted recommendation:**

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

### Response Codes

| Status | Description |
|---|---|
| **200** | Always HTTP 200. Inspect `status` field. |
| **401** | Unauthorised. |
| **404** | Condition not found. |

### TypeScript Example

```typescript
const result = await client.conditions.calibrate({
  conditionId: "org.high_churn",
  conditionVersion: "1.0",
  target: { alertsPerDay: 5 },
});

if (result.status === "recommendation_available") {
  console.log(result.recommended_params); // { value: 0.85 }
  console.log(result.impact.delta_alerts); // -12
} else {
  console.log(result.no_recommendation_reason);
}
```

---

## Apply Calibration

```
POST /conditions/apply-calibration
```

Creates a new immutable condition version from the calibration recommendation. The existing version is never modified.

:::warning Rebinding is Always Explicit
This endpoint does **not** auto-rebind tasks. It returns `tasks_pending_rebind` so you know which tasks still need rebinding. Call `PATCH /tasks/&#123;id&#125;` for each. This prevents silent behavior changes.
:::

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `calibration_token` | string | **Required** | Token from `POST /conditions/calibrate`. Single-use, expires 24h. |
| `new_version` | string | Optional | Version string. Auto-incremented if not provided. |

### Response — ApplyCalibrationResult

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_id` | string | Always | Condition identifier. |
| `previous_version` / `new_version` | string | Always | The calibrated and newly created versions. |
| `params_applied` | object | Always | Parameters written to the new version. |
| `tasks_pending_rebind` | array | Always | Tasks still bound to `previous_version`. Each item: `task_id`, `intent`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | New condition version created. |
| **400** | Invalid or expired token. |
| **401** | Unauthorised. |

### TypeScript Example

```typescript
const applied = await client.conditions.applyCalibration({
  calibrationToken: result.calibration_token,
});

console.log(applied.new_version);    // "1.1"
console.log(applied.params_applied); // { value: 0.85 }

for (const t of applied.tasks_pending_rebind) {
  await client.tasks.update(t.task_id, { conditionVersion: applied.new_version });
}
```
