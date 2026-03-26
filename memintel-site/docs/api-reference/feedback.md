---
id: feedback
title: Feedback
sidebar_label: Feedback
---

# Feedback

Records signal about whether a Decision (Aₜ) was correct. Stored feedback is consumed by `POST /conditions/calibrate` to derive the tighten or relax direction automatically.

---

## Submit Decision Feedback

```
POST /feedback/decision
```

Records a feedback signal on a specific Decision (Aₜ).

| Value | Meaning | Calibration Effect |
|---|---|---|
| `false_positive` | Condition fired but should not have. | Calibration will **tighten**. |
| `false_negative` | Condition did not fire but should have. | Calibration will **relax**. |
| `correct` | Decision was expected. | No-op (stored for audit). |

:::note Feedback and the Equals Strategy
Feedback is stored for all strategies. But `POST /conditions/calibrate` always returns `no_recommendation` for `equals` conditions — feedback cannot drive numeric parameter adjustment for categorical conditions.
:::

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_id` | string | **Required** | Fully qualified condition identifier. |
| `condition_version` | string | **Required** | Explicit condition version. |
| `entity` | string | **Required** | Entity the Decision (Aₜ) was made for. |
| `timestamp` | datetime | **Required** | ISO 8601 UTC timestamp of the decision being rated. |
| `feedback` | enum | **Required** | `false_positive` \| `false_negative` \| `correct`. |
| `note` | string | Optional | Free-text audit note. |

### Response

| Parameter | Type | Required | Description |
|---|---|---|---|
| `status` | enum | Always | `recorded`. |
| `feedback_id` | string | Always | Unique feedback record identifier. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Feedback recorded. |
| **400** | Invalid value or missing field. |
| **401** | Unauthorised. |
| **404** | Condition not found. |

### TypeScript Example

```typescript
await client.feedback.decision({
  conditionId: "org.high_churn",
  conditionVersion: "1.0",
  entity: "user_abc123",
  timestamp: "2024-03-15T09:00:00Z",
  feedback: "false_positive",
  note: "One-off spike — user was active.",
});

// Now calibrate — direction derived automatically from stored feedback
const cal = await client.conditions.calibrate({
  conditionId: "org.high_churn",
  conditionVersion: "1.0",
});
```
