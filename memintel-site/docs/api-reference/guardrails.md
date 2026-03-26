---
id: guardrails
title: Guardrails
sidebar_label: Guardrails
---

# Guardrails

Guardrails define the policy layer that constrains what the compiler can generate — which evaluation strategies are permitted, what parameter ranges are valid, and how severity language maps to numeric thresholds.

Guardrails can be managed in two ways:
- **Via API** (recommended) — takes effect immediately, no server restart required, fully versioned
- **Via file** (`memintel_guardrails.yaml`) — requires server restart, no version history

Once you post guardrails via API, the API version always takes precedence over the file.

:::tip
Use the API for all guardrails changes after initial deployment. It is simpler, takes effect immediately, and maintains a full audit trail.
:::

---

## POST /guardrails

Create a new guardrails version. Deactivates the previous version atomically. Takes effect immediately — no server restart required.

:::warning Elevated key required
This endpoint requires the `X-Elevated-Key` header set to the value of `MEMINTEL_ELEVATED_KEY`. All GET endpoints do not require the elevated key.
:::

```bash
curl -X POST https://your-domain/guardrails \
  -H "Content-Type: application/json" \
  -H "X-Elevated-Key: your-elevated-key" \
  -d @guardrails.json
```

### Request Schema

| Field | Type | Required | Description |
|---|---|---|---|
| `guardrails.strategy_registry` | list[string] | Yes | Permitted evaluation strategies. Valid values: `threshold`, `percentile`, `z_score`, `change`, `equals`, `composite` |
| `guardrails.type_strategy_map` | object | Yes | Maps signal types to permitted strategies. Keys: `float`, `int`, `boolean`, `string`, `categorical`, `time_series<float>`, `time_series<int>`, `float?`, `int?` |
| `guardrails.parameter_priors` | object | No | Per-signal threshold priors. Each signal has `low_severity`, `medium_severity`, `high_severity` with `value` and optional `window` |
| `guardrails.bias_rules` | object | No | Maps natural language severity words to severity levels. Valid levels: `high_severity`, `medium_severity`, `low_severity` |
| `guardrails.threshold_directions` | object | No | Maps signal IDs to `above` or `below`. Default is `above` if not specified |
| `guardrails.global_preferred_strategy` | string | No | Preferred strategy when multiple are valid. Default: `percentile` |
| `guardrails.global_default_strategy` | string | No | Fallback strategy when no other rule matches. Default: `threshold` |
| `change_note` | string \| null | No | Human-readable reason for this change. Stored for audit purposes |

### Validation

The endpoint validates the definition before saving. Violations return HTTP `400` with `error.type: semantic_error`:

- All strategies in `strategy_registry` must be known valid strategies
- `type_strategy_map` entries must only reference strategies in `strategy_registry`
- `bias_rules` values must be exactly: `high_severity`, `medium_severity`, or `low_severity`
- `global_preferred_strategy` must be in `strategy_registry`
- `global_default_strategy` must be in `strategy_registry`

### Example Request

```json
{
  "guardrails": {
    "strategy_registry": [
      "threshold", "percentile", "z_score", "change", "equals", "composite"
    ],
    "type_strategy_map": {
      "float":               ["threshold", "percentile", "z_score", "change"],
      "int":                 ["threshold", "percentile", "change"],
      "boolean":             ["equals"],
      "categorical":         ["equals"],
      "time_series<float>":  ["z_score", "change", "percentile"],
      "float?":              ["threshold"]
    },
    "parameter_priors": {
      "account.active_user_rate_30d": {
        "low_severity":    { "value": 0.60 },
        "medium_severity": { "value": 0.45 },
        "high_severity":   { "value": 0.30 }
      }
    },
    "bias_rules": {
      "urgent":      "high_severity",
      "significant": "medium_severity",
      "early":       "low_severity"
    },
    "threshold_directions": {
      "account.active_user_rate_30d": "below"
    },
    "global_preferred_strategy": "percentile",
    "global_default_strategy":   "threshold"
  },
  "change_note": "Initial guardrails for SaaS churn detection domain"
}
```

### Response Schema — GuardrailsVersion

| Field | Type | Description |
|---|---|---|
| `guardrails_id` | string (UUID) | Auto-generated unique identifier |
| `version` | string | Auto-assigned version string: `v1`, `v2`, `v3`... |
| `guardrails` | object | The full guardrails definition as submitted |
| `change_note` | string \| null | The change note as submitted |
| `created_at` | datetime | ISO 8601 UTC timestamp |
| `is_active` | boolean | `true` if this is the currently active version |
| `source` | string | `api` for API-created versions. `file` for startup-loaded versions |

### Example Response

```json
{
  "guardrails_id": "grls_8f3k2m...",
  "version": "v1",
  "is_active": true,
  "source": "api",
  "change_note": "Initial guardrails for SaaS churn detection domain",
  "created_at": "2026-03-27T10:00:00Z",
  "guardrails": { "..." : "..." }
}
```

### Response Codes

| Code | Description |
|---|---|
| `201` | Guardrails created. New version active immediately. |
| `400` | Semantic validation failed. `error.type: semantic_error` with descriptive message. |
| `401` | Missing or invalid API key. |
| `403` | Missing or invalid elevated key. `X-Elevated-Key` header required. |

---

## GET /guardrails

Get the currently active guardrails version.

:::note
When no guardrails have been posted via API, this endpoint returns HTTP `404`:
- `error.type: not_found`
- Message: `"No guardrails defined via API. Guardrails loaded from memintel_guardrails.yaml at startup."`

This is expected behaviour — not an error. It means the deployment is using file-based guardrails, which is valid.
:::

### Response Codes

| Code | Description |
|---|---|
| `200` | Active guardrails version returned |
| `404` | No API guardrails defined. File-based guardrails in use. |

---

## GET /guardrails/versions

List all guardrails versions, newest first.

**Response:** Array of GuardrailsVersion objects ordered by `created_at` descending.

---

## GET /guardrails/versions/&#123;version&#125;

Get a specific guardrails version by version string.

| Parameter | Type | Description |
|---|---|---|
| `version` | string | Version string — e.g. `v1`, `v2`, `v3` |

### Response Codes

| Code | Description |
|---|---|
| `200` | Guardrails version returned |
| `404` | Version not found |

---

## GET /guardrails/impact

Shows how many tasks were compiled under older guardrails versions — useful for identifying tasks that may benefit from recompilation under current guardrails.

### Response Schema

| Field | Type | Description |
|---|---|---|
| `current_version` | string | The currently active guardrails version |
| `tasks_on_current_version` | int | Tasks compiled under the current version |
| `tasks_on_older_versions` | array | List of `{ version, task_count }` for each older version with active tasks |
| `total_stale_tasks` | int | Total tasks not on the current guardrails version |

---

## File vs API Precedence

Understanding which guardrails are used in each scenario:

| Scenario | Guardrails in use |
|---|---|
| Fresh deployment, no `POST /guardrails` called yet | `memintel_guardrails.yaml` loaded from file at startup |
| `POST /guardrails` called at least once | Most recent API version — file is ignored |
| `POST /guardrails` called, then server restarted | API version reloaded from DB — file still ignored |
| API version deleted from DB (manual intervention) | Falls back to file at next restart |

:::tip
The file is the seed and fallback. The API is the override. Once you post via API, the API version always wins.
:::

---

## Immediate Effect

```
POST /guardrails
      ↓
Server validates the definition
      ↓
New version written to database
      ↓
Guardrails reloaded into memory
      ↓
201 response returned
```

Changes take effect **before** the response is returned. Any task created after this point uses the new guardrails version.

---

## Task Provenance

Every task records which guardrails version was active when it was compiled. The POST /tasks response includes three provenance fields:

| Field | Type | Description |
|---|---|---|
| `context_version` | string \| null | Context version active at task creation. Null if no context defined. |
| `guardrails_version` | string \| null | Guardrails version active at task creation. Null if file-based guardrails were in use (no API version posted yet). |
| `context_warning` | string \| null | Warning if no application context was defined. Null if context was present. |
