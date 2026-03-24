---
id: decisions
title: Decisions
sidebar_label: Decisions
---

# Decisions

**Instance-level.** Explains a specific Decision (Aₜ) for a given entity at a given timestamp. Deterministic: the same inputs always produce the same explanation.

---

## Explain Decision

```
POST /decisions/explain
```

Returns a full explanation of a Decision (Aₜ): the Result (Rₜ) that was evaluated, the strategy and parameters applied, whether the condition fired, and the contribution of each input signal.

**Boolean strategies** (`threshold`, `percentile`, `change`, `z_score`, `composite`): `decision` is `true`/`false`. `threshold_applied` contains the numeric cutoff compared against the Result (Rₜ).

**Equals strategy:** `decision` is the matched label string. `label_matched` contains it explicitly. `threshold_applied` is `null`.

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_id` | string | **Required** | Fully qualified condition identifier. |
| `condition_version` | string | **Required** | Explicit condition version. |
| `entity` | string | **Required** | Entity the Decision (Aₜ) was made for. |
| `timestamp` | datetime | **Required** | ISO 8601 UTC. Retrieves the stored Decision (Aₜ) at this timestamp. |

### Response — DecisionExplanation

| Parameter | Type | Required | Description |
|---|---|---|---|
| `decision_type` | enum | Always | `boolean` \| `categorical`. |
| `decision` | boolean\|string | Always | Decision (Aₜ) value: `true`/`false` or matched label. |
| `concept_value` | number\|string | Always | The Result (Rₜ) that was evaluated. |
| `strategy_type` | enum | Always | `threshold` \| `percentile` \| `change` \| `z_score` \| `equals` \| `composite`. |
| `threshold_applied` | number | Conditional | Numeric cutoff. Present for `threshold`, `percentile`, `change`, `z_score`. `null` for `equals`, `composite`. |
| `label_matched` | string | Conditional | Matched label for `equals` strategy. `null` if no match. |
| `drivers` | array | Always | `signal`, `contribution`, `value` per input signal. |

### Response Codes

| Status | Description |
|---|---|
| **200** | DecisionExplanation. |
| **401** | Unauthorised. |
| **404** | Decision not found. |

### TypeScript Example

```typescript
// Boolean decision
const exp = await client.decisions.explain({
  conditionId: "org.high_churn",
  conditionVersion: "1.0",
  entity: "user_abc123",
  timestamp: "2024-03-15T09:00:00Z",
});

console.log(exp.decision_type);      // "boolean"
console.log(exp.decision);           // true  (Aₜ)
console.log(exp.concept_value);      // 0.87  (Rₜ)
console.log(exp.threshold_applied);  // 0.8

// Categorical decision (equals strategy)
const cat = await client.decisions.explain({
  conditionId: "org.segment_check",
  conditionVersion: "1.0",
  entity: "user_abc123",
  timestamp: "2024-03-15T09:00:00Z",
});

console.log(cat.decision_type);      // "categorical"
console.log(cat.decision);           // "high_risk"
console.log(cat.threshold_applied);  // null
```
