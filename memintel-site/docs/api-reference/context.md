---
id: context
title: Application Context
sidebar_label: Application Context
---

# Application Context

Application context is an optional but strongly recommended configuration layer that gives the LLM domain knowledge before any tasks are compiled. It affects the accuracy of all task definitions and calibration recommendations.

---

## POST /context

Create or update application context. Creates a new immutable version and deactivates the previous one atomically. Takes effect immediately.

### Request Schema

| Field | Type | Required | Description |
|---|---|---|---|
| `domain.description` | string | Yes | Natural language description of what the application does and what domain it operates in |
| `domain.entities` | array | No | List of entity declarations. Each has `name` (string) and `description` (string) |
| `domain.decisions` | array | No | List of decision type names (e.g. `churn_risk`, `fraud_alert`) |
| `behavioural.data_cadence` | enum | No | `batch` \| `streaming` \| `mixed`. Default: `batch` |
| `behavioural.meaningful_windows` | object | No | `min` and `max` duration strings (e.g. `30d`, `90d`). Informs calibration window clamping |
| `behavioural.regulatory` | array | No | Regulatory frameworks in scope (e.g. `GDPR`, `SOC2`, `HIPAA`) |
| `semantic_hints` | array | No | Domain-specific term definitions. Each has `term` (string) and `definition` (string) |
| `calibration_bias.false_negative_cost` | enum | No | `high` \| `medium` \| `low` |
| `calibration_bias.false_positive_cost` | enum | No | `high` \| `medium` \| `low` |

`bias_direction` is auto-derived and cannot be set manually: `false_negative_cost > false_positive_cost` → `recall`. Reverse → `precision`. Equal → `balanced`.

### Example Request

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market software companies.",
    "entities": [
      { "name": "account", "description": "company-level subscription" },
      { "name": "user",    "description": "individual platform user" }
    ],
    "decisions": ["churn_risk", "expansion_opportunity"]
  },
  "behavioural": {
    "data_cadence": "batch",
    "meaningful_windows": { "min": "30d", "max": "90d" },
    "regulatory": ["GDPR", "SOC2"]
  },
  "semantic_hints": [
    { "term": "active user",        "definition": "logged in AND performed core action in last 14 days" },
    { "term": "high value account", "definition": "ARR above $50,000" }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium"
  }
}
```

### Response Schema — ApplicationContext

| Field | Type | Description |
|---|---|---|
| `context_id` | string (UUID) | Auto-generated unique identifier |
| `version` | string | Auto-assigned: `v1`, `v2`, `v3`... |
| `domain` | object | The domain context as submitted |
| `behavioural` | object | The behavioural context as submitted |
| `semantic_hints` | array | The semantic hints as submitted |
| `calibration_bias` | object \| null | The calibration bias including auto-derived `bias_direction` |
| `created_at` | datetime | ISO 8601 UTC timestamp |
| `is_active` | boolean | `true` if this is the currently active version |

### Response Codes

| Code | Description |
|---|---|
| `201` | Context created. New version active immediately. |
| `400` | Invalid request — missing required fields or invalid enum values |
| `401` | Unauthorised — invalid or missing API key |

---

## GET /context

Get the currently active context version.

:::note
When no context has been defined, returns HTTP `404`:
- `error.type: not_found`
- Message: `"No active application context exists."`

This is expected behaviour — not an error. It means context has not been configured yet.
:::

### Response Codes

| Code | Description |
|---|---|
| `200` | Active context returned |
| `404` | No active context defined |

---

## GET /context/versions

List all context versions, newest first.

**Response:** Array of ApplicationContext objects ordered by `created_at` descending.

---

## GET /context/versions/&#123;version&#125;

Get a specific context version by version string.

| Parameter | Type | Description |
|---|---|---|
| `version` | string | Version string — e.g. `v1`, `v2`, `v3` |

### Response Codes

| Code | Description |
|---|---|
| `200` | Context version returned |
| `404` | Version not found |

---

## GET /context/impact

Shows how many tasks were compiled under older context versions.

### Response Schema

| Field | Type | Description |
|---|---|---|
| `current_version` | string | The currently active context version |
| `tasks_on_current_version` | int | Tasks compiled under the current version |
| `tasks_on_older_versions` | array | List of `{ version, task_count }` for each older version |
| `total_stale_tasks` | int | Total tasks not on the current context version |

---

## Effect on Task Creation

When context is active, the POST /tasks response includes:

| Field | Type | Description |
|---|---|---|
| `context_version` | string \| null | Context version active at task creation. `null` if no context defined. |
| `context_warning` | string \| null | `"No active application context exists..."` if no context was defined. `null` if context was present. |

---

## Effect on Calibration

When `calibration_bias` is defined, `POST /conditions/calibrate` returns additional fields:

| Field | Type | Description |
|---|---|---|
| `statistically_optimal` | float | Raw optimal value from feedback data alone |
| `context_adjusted` | float \| null | Bias-adjusted value after applying `calibration_bias`. `null` if no bias defined. |
| `recommended` | float | Final recommended value — equals `context_adjusted` if bias was applied |
| `adjustment_explanation` | string \| null | Human-readable explanation of any adjustment |
