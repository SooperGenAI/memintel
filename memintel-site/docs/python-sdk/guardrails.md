---
id: guardrails
title: Guardrails
sidebar_label: Guardrails
---

# Guardrails

Endpoints for managing the guardrails policy via API. Guardrails define which evaluation strategies are permitted, what parameter ranges are valid, and how severity language maps to numeric thresholds. Changes via API take effect immediately — no server restart required.

:::note File-based alternative
Guardrails can also be managed via `memintel_guardrails.yaml` loaded at startup. Once any version is posted via API, the API version always takes precedence over the file. See the [Admin Guide](/docs/admin-guide/admin-guardrails-api) for details.
:::

---

## Create Guardrails Version

```
POST /guardrails
```

Creates a new guardrails version and immediately activates it. Deactivates the previous version atomically. Validates the definition before saving — returns HTTP 400 on semantic errors.

:::warning Elevated key required
This endpoint requires the `X-Elevated-Key` header in addition to the standard `X-API-Key`. The elevated key is the value of `MEMINTEL_ELEVATED_KEY` set in your server environment.
:::

### Request — CreateGuardrailsRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `guardrails` | dict | **Required** | The guardrails definition. See sub-fields below. |
| `guardrails.strategy_registry` | list[str] | **Required** | Permitted evaluation strategies. Valid values: `threshold`, `percentile`, `z_score`, `change`, `equals`, `composite`. |
| `guardrails.type_strategy_map` | dict | **Required** | Maps signal types to permitted strategies. Keys: `float`, `int`, `boolean`, `string`, `categorical`, `time_series<float>`, `time_series<int>`, `float?`, `int?`. |
| `guardrails.parameter_priors` | dict | Optional | Per-signal threshold priors. Each signal maps to `{ low_severity, medium_severity, high_severity }` with `value` and optional `window`. |
| `guardrails.bias_rules` | dict | Optional | Maps natural language severity words to severity levels (`high_severity`, `medium_severity`, `low_severity`). |
| `guardrails.threshold_directions` | dict | Optional | Maps signal IDs to `'above'` or `'below'`. Default is `'above'` if not specified. |
| `guardrails.global_preferred_strategy` | str | Optional | Preferred strategy when multiple are valid. Default: `'percentile'`. |
| `guardrails.global_default_strategy` | str | Optional | Fallback when no other rule matches. Default: `'threshold'`. |
| `change_note` | str \| None | Optional | Human-readable reason for this change. Stored for audit purposes. |

### Validation Rules

The endpoint validates before saving. HTTP 400 with `error.type: semantic_error` if:

- Any strategy in `strategy_registry` is not a known valid strategy
- `type_strategy_map` references a strategy not in `strategy_registry`
- `bias_rules` values are not exactly `high_severity`, `medium_severity`, or `low_severity`
- `global_preferred_strategy` or `global_default_strategy` is not in `strategy_registry`

### Response — GuardrailsVersion

| Field | Type | Description |
|---|---|---|
| `guardrails_id` | str (UUID) | Auto-generated unique identifier. |
| `version` | str | Auto-assigned: `v1`, `v2`, `v3`... |
| `guardrails` | dict | The full guardrails definition as submitted. |
| `change_note` | str \| None | The change note as submitted. |
| `created_at` | str | ISO 8601 UTC timestamp. |
| `is_active` | bool | `True` — newly created versions are always immediately active. |
| `source` | str | Always `'api'` for API-created versions. `'file'` for startup-loaded versions. |

### Response Codes

| Status | Description |
|---|---|
| **201** | Guardrails created and active. Takes effect immediately. |
| **400** | Semantic validation failed. `error.type: semantic_error` with descriptive message. |
| **401** | Unauthorised — missing or invalid `X-API-Key`. |
| **403** | Missing or invalid elevated key. `X-Elevated-Key` header required. |

### Python Example

```python
import memintel

client = memintel.AsyncClient()

guardrails = await client.guardrails.create(
    guardrails={
        "strategy_registry": [
            "threshold", "percentile", "z_score", "change", "equals", "composite"
        ],
        "type_strategy_map": {
            "float":              ["threshold", "percentile", "z_score", "change"],
            "int":                ["threshold", "percentile", "change"],
            "boolean":            ["equals"],
            "categorical":        ["equals"],
            "time_series<float>": ["z_score", "change", "percentile"],
            "float?":             ["threshold"],
        },
        "parameter_priors": {
            "account.active_user_rate_30d": {
                "low_severity":    {"value": 0.60},
                "medium_severity": {"value": 0.45},
                "high_severity":   {"value": 0.30},
            },
            "account.days_to_renewal": {
                "low_severity":    {"value": 90},
                "medium_severity": {"value": 60},
                "high_severity":   {"value": 30},
            },
        },
        "bias_rules": {
            "urgent":      "high_severity",
            "significant": "medium_severity",
            "early":       "low_severity",
            "approaching": "low_severity",
        },
        "threshold_directions": {
            "account.active_user_rate_30d": "below",
            "account.days_to_renewal":      "below",
        },
        "global_preferred_strategy": "percentile",
        "global_default_strategy":   "threshold",
    },
    change_note="Initial guardrails — SaaS churn detection domain",
    elevated_key="your-elevated-key",  # or set MEMINTEL_ELEVATED_KEY env var
)

print(guardrails.version)    # v1
print(guardrails.is_active)  # True
print(guardrails.source)     # api
```

---

## Get Active Guardrails

```
GET /guardrails
```

Returns the currently active guardrails version. Returns HTTP 404 if no guardrails have been posted via API — meaning file-based guardrails are in use.

### Response Codes

| Status | Description |
|---|---|
| **200** | Active GuardrailsVersion returned. |
| **401** | Unauthorised. |
| **404** | No API guardrails defined. File-based guardrails in use. `error.type: not_found`. |

### Python Example

```python
try:
    g = await client.guardrails.get_active()
    print(g.version)   # v2
    print(g.source)    # api
except memintel.NotFoundError:
    print("No API guardrails — file-based guardrails in use")
```

---

## List Guardrails Versions

```
GET /guardrails/versions
```

Returns all guardrails versions, newest first. Includes both API-created (`source: api`) and startup-loaded (`source: file`) versions.

### Response

`list[GuardrailsVersion]` ordered by `created_at` descending.

### Python Example

```python
versions = await client.guardrails.list_versions()

for v in versions:
    print(f"{v.version} — {v.source} — {'active' if v.is_active else 'inactive'}")
    if v.change_note:
        print(f"  Note: {v.change_note}")
# v2 — api  — active   — Note: Tightened AML thresholds per new FATF guidance
# v1 — api  — inactive — Note: Initial guardrails
```

---

## Get Guardrails Version

```
GET /guardrails/versions/{version}
```

Returns a specific guardrails version by version string.

### Path Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | str | **Required** | Version string — e.g. `'v1'`, `'v2'`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | GuardrailsVersion for the requested version. |
| **401** | Unauthorised. |
| **404** | Version not found. |

### Python Example

```python
v1 = await client.guardrails.get_version("v1")
print(v1.guardrails["bias_rules"])
print(v1.change_note)
print(v1.is_active)  # False — superseded
```

---

## Get Guardrails Impact

```
GET /guardrails/impact
```

Returns a summary of how many tasks were compiled under older guardrails versions. Use after updating guardrails to identify tasks that would benefit from recompilation under the new policy.

### Response — GuardrailsImpactResult

| Field | Type | Description |
|---|---|---|
| `current_version` | str | The currently active guardrails version. |
| `tasks_on_current_version` | int | Tasks compiled under the current version. |
| `tasks_on_older_versions` | list[dict] | List of `{ version, task_count }` for each older version with active tasks. |
| `total_stale_tasks` | int | Total tasks not on the current guardrails version. |

### Response Codes

| Status | Description |
|---|---|
| **200** | GuardrailsImpactResult. |
| **401** | Unauthorised. |

### Python Example

```python
impact = await client.guardrails.get_impact()

print(f"Current version: {impact.current_version}")
print(f"Tasks on current: {impact.tasks_on_current_version}")
print(f"Stale tasks: {impact.total_stale_tasks}")

for entry in impact.tasks_on_older_versions:
    print(f"  {entry['version']}: {entry['task_count']} tasks on older guardrails")

# Trigger recompilation of stale tasks if needed
# (via client.tasks.recompile or PATCH /tasks/{id})
```

---

## File vs API Precedence

| Scenario | Guardrails in use |
|---|---|
| No `POST /guardrails` called yet | `memintel_guardrails.yaml` loaded from file at startup |
| `POST /guardrails` called at least once | Most recent API version — file is ignored |
| Server restarted after API post | API version reloaded from database — file still ignored |

The file is the seed and fallback. The API is the override. Once any version is posted via API, `GET /guardrails` will always return the API version.
